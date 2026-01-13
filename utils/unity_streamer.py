import socket
import json
import threading
import time
import torch
import numpy as np
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R

# SMPL Joint Names (22 joints)
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
]

# Mapping from SMPL index to Unity Humanoid Bone Name
# This mapping assumes a standard Unity Mecanim Humanoid setup.
# Note: Unity often handles parent-child retargeting, but we map logic here.
SMPL_TO_UNITY_MAP = {
    0: "Hips",
    1: "LeftUpperLeg",
    2: "RightUpperLeg",
    3: "Spine",
    4: "LeftLowerLeg",
    5: "RightLowerLeg",
    6: "Chest",       # Spine2
    7: "LeftFoot",
    8: "RightFoot",
    9: "UpperChest",  # Spine3
    10: "LeftToes",
    11: "RightToes",
    12: "Neck",
    13: "LeftShoulder",
    14: "RightShoulder",
    15: "Head",
    16: "LeftUpperArm",
    17: "RightUpperArm",
    18: "LeftLowerArm",
    19: "RightLowerArm",
    20: "LeftHand",
    21: "RightHand"
}

class UnityStreamer:
    def __init__(self, host='127.0.0.1', port=8080):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.lock = threading.Lock()
        
        self._start_server_thread()

    def _start_server_thread(self):
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

    def _server_loop(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"[UnityStreamer] Listening on {self.host}:{self.port}...")
            self.is_running = True
            
            while self.is_running:
                try:
                    client, addr = self.server_socket.accept()
                    print(f"[UnityStreamer] Client connected: {addr}")
                    with self.lock:
                        if self.client_socket:
                            try:
                                self.client_socket.close()
                            except:
                                pass
                        self.client_socket = client
                except OSError:
                    break
        except Exception as e:
            print(f"[UnityStreamer] Server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def close(self):
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        if self.client_socket:
            self.client_socket.close()

    def send_frame(self, transl, global_orient, body_pose):
        """
        Send a frame of animation data.
        Args:
            transl: (3,) tensor or array, root position
            global_orient: (3, 3) tensor or (1, 3, 3), root rotation matrix
            body_pose: (21, 3, 3) tensor, body joint rotation matrices
        """
        with self.lock:
            if not self.client_socket:
                return

            try:
                # Prepare data
                data = self._process_frame(transl, global_orient, body_pose)
                json_data = json.dumps(data) + "\n" # Newline delimiter for Unity
                self.client_socket.sendall(json_data.encode('utf-8'))
            except (ConnectionResetError, BrokenPipeError):
                print("[UnityStreamer] Client disconnected")
                self.client_socket = None
            except Exception as e:
                print(f"[UnityStreamer] Send error: {e}")

    def _process_frame(self, transl, global_orient, body_pose):
        # Convert tensors to numpy if needed
        if isinstance(transl, torch.Tensor):
            transl = transl.detach().cpu().numpy().flatten()
        if isinstance(global_orient, torch.Tensor):
             global_orient = global_orient.detach().cpu().numpy()
        if isinstance(body_pose, torch.Tensor):
            body_pose = body_pose.detach().cpu().numpy()

        # Ensure shapes
        global_orient = global_orient.reshape(3, 3)
        body_pose = body_pose.reshape(21, 3, 3)

        # 1. Convert Root Position (Right-Handed -> Left-Handed)
        # Unity: +X Right, +Y Up, +Z Forward
        # SMPL/PyTorch3D: +X Left?, +Y Up, +Z Forward? 
        # Usually: x -> -x
        root_pos = [float(-transl[0]), float(transl[1]), float(transl[2])]

        # 2. Convert Rotations to Quaternions
        # Combine global_orient and body_pose
        # global_orient is index 0
        all_rots = [global_orient] + [body_pose[i] for i in range(21)]
        
        joints_list = []
        for i, rot_mat in enumerate(all_rots):
            if i not in SMPL_TO_UNITY_MAP:
                continue
            
            # Matrix to Quat (Scipy uses x, y, z, w scalar-last)
            r = R.from_matrix(rot_mat)
            quat = r.as_quat() # x, y, z, w
            
            # Coordinate System Conversion for Rotation (RH -> LH)
            # Typically: (x, y, z, w) -> (-x, y, z, -w) or (x, -y, -z, w)
            # DART generally uses: x right, y up, z forward?
            # Let's try standard conversion: -x, -y, z, w ? 
            # Actually, standard conversion from OpenGL (RH) to Unity (LH) for Quats is:
            # (-x, -y, z, w) if converting basis 
            # But here we are mirroring X axis.
            # Let's use (-x, y, z, -w) which is common for mirroring X.
            
            qx, qy, qz, qw = quat
            
            # Simple conversion that often works for mirrored x-axis:
            # Ensure float conversion
            unity_quat = [float(-qx), float(-qy), float(qz), float(qw)]
            
            name = SMPL_TO_UNITY_MAP[i]
            joints_list.append({
                "name": name,
                "rot": unity_quat # [x, y, z, w]
            })

        return {
            "root_pos": root_pos,
            "joints": joints_list
        }
