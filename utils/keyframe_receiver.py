"""
Unity Keyframe Receiver - receives keyframe data for motion in-betweening from Unity via TCP
"""
import socket
import json
import threading
import queue
from typing import Callable, Optional, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation as R


class InbetweenRequest:
    """Data class for an in-betweening request"""
    def __init__(self, data: dict):
        self.start_frame = data.get('start_frame', {})
        self.end_frame = data.get('end_frame', {})
        self.prompt = data.get('prompt', 'walk')
        self.duration_frames = data.get('duration_frames', 60)
        self.mode = data.get('mode', 'fast')  # 'fast' or 'quality'
        
        # Convert Unity keyframes to SMPL format
        self.start_smpl = self._unity_to_smpl(self.start_frame)
        self.end_smpl = self._unity_to_smpl(self.end_frame)
    
    def _unity_to_smpl(self, frame: dict) -> dict:
        """Convert Unity Humanoid frame to SMPL parameters"""
        if not frame:
            return None
            
        # Unity to SMPL joint mapping (reverse of what's in unity_streamer.py)
        unity_to_smpl_map = {
            "Hips": 0,
            "LeftUpperLeg": 1,
            "RightUpperLeg": 2,
            "Spine": 3,
            "LeftLowerLeg": 4,
            "RightLowerLeg": 5,
            "Chest": 6,
            "LeftFoot": 7,
            "RightFoot": 8,
            "UpperChest": 9,
            "LeftToes": 10,
            "RightToes": 11,
            "Neck": 12,
            "LeftShoulder": 13,
            "RightShoulder": 14,
            "Head": 15,
            "LeftUpperArm": 16,
            "RightUpperArm": 17,
            "LeftLowerArm": 18,
            "RightLowerArm": 19,
            "LeftHand": 20,
            "RightHand": 21
        }
        
        # Convert root position (Unity LH -> SMPL RH)
        root_pos = frame.get('root_pos', [0, 0, 0])
        # Reverse the conversion done in UnityStreamer: x -> -x
        smpl_transl = np.array([-root_pos[0], root_pos[1], root_pos[2]], dtype=np.float32)
        
        # Initialize rotation matrices (22 joints)
        # Index 0 is global_orient, indices 1-21 are body_pose
        rotations = [np.eye(3, dtype=np.float32) for _ in range(22)]
        
        joints = frame.get('joints', [])
        for joint in joints:
            name = joint.get('name', '')
            if name not in unity_to_smpl_map:
                continue
            
            idx = unity_to_smpl_map[name]
            quat = joint.get('rot', [0, 0, 0, 1])  # [x, y, z, w]
            
            # Reverse the quaternion conversion from UnityStreamer
            # UnityStreamer did: (-qx, -qy, qz, qw)
            # So we reverse: (-qx, -qy, qz, qw) -> (qx, qy, qz, qw) with proper sign
            qx, qy, qz, qw = quat
            smpl_quat = [-qx, -qy, qz, qw]  # Reverse the Unity conversion
            
            # Convert quaternion to rotation matrix
            r = R.from_quat(smpl_quat)  # scipy uses [x, y, z, w] order
            rotations[idx] = r.as_matrix().astype(np.float32)
        
        return {
            'transl': smpl_transl,
            'global_orient': rotations[0],
            'body_pose': np.array(rotations[1:], dtype=np.float32),  # [21, 3, 3]
        }
    
    def __repr__(self):
        return f"InbetweenRequest(prompt='{self.prompt}', duration={self.duration_frames}, mode='{self.mode}')"


class UnityKeyframeReceiver:
    """
    TCP server that receives keyframe data and in-betweening requests from Unity.
    Runs on a separate port from motion streaming (default: 8082).
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8082,
                 on_request_received: Optional[Callable[[InbetweenRequest], None]] = None):
        """
        Args:
            host: Host to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on (default 8082)
            on_request_received: Callback function called when a request is received
        """
        self.host = host
        self.port = port
        self.on_request_received = on_request_received
        
        self.server_socket = None
        self.client_socket = None
        self.is_running = False
        self.request_queue = queue.Queue()
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
            print(f"[UnityKeyframeReceiver] Listening for keyframe requests on {self.host}:{self.port}...")
            self.is_running = True
            
            while self.is_running:
                try:
                    client, addr = self.server_socket.accept()
                    print(f"[UnityKeyframeReceiver] Unity client connected: {addr}")
                    
                    with self.lock:
                        if self.client_socket:
                            try:
                                self.client_socket.close()
                            except:
                                pass
                        self.client_socket = client
                    
                    # Handle this client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,),
                        daemon=True
                    )
                    client_thread.start()
                    
                except OSError:
                    break
                    
        except Exception as e:
            print(f"[UnityKeyframeReceiver] Server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def _handle_client(self, client_socket):
        """Handle incoming messages from a connected client"""
        buffer = ""
        
        try:
            while self.is_running:
                data = client_socket.recv(8192)  # Larger buffer for keyframe data
                if not data:
                    break
                
                buffer += data.decode('utf-8')
                
                # Process complete messages (newline-delimited JSON)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        self._process_message(line, client_socket)
                        
        except (ConnectionResetError, BrokenPipeError):
            print("[UnityKeyframeReceiver] Client disconnected")
        except Exception as e:
            print(f"[UnityKeyframeReceiver] Client error: {e}")
        finally:
            client_socket.close()
    
    def _process_message(self, message: str, client_socket):
        """Process a received JSON message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            if msg_type == 'inbetween_request':
                request = InbetweenRequest(data)
                print(f"[UnityKeyframeReceiver] Received: {request}")
                
                # Add to queue
                self.request_queue.put(request)
                
                # Send acknowledgment
                self._send_response(client_socket, {
                    'type': 'inbetween_ack',
                    'status': 'processing',
                    'duration_frames': request.duration_frames
                })
                
                # Call callback if set
                if self.on_request_received:
                    self.on_request_received(request)
                    
            elif msg_type == 'ping':
                self._send_response(client_socket, {'type': 'pong'})
                    
        except json.JSONDecodeError as e:
            print(f"[UnityKeyframeReceiver] Invalid JSON: {e}")
            self._send_response(client_socket, {
                'type': 'error',
                'message': f'Invalid JSON: {e}'
            })
    
    def _send_response(self, client_socket, response: dict):
        """Send a response to the client"""
        try:
            json_data = json.dumps(response) + "\n"
            client_socket.sendall(json_data.encode('utf-8'))
        except Exception as e:
            print(f"[UnityKeyframeReceiver] Send error: {e}")
    
    def send_status(self, status: dict):
        """Send status update to connected client"""
        with self.lock:
            if self.client_socket:
                try:
                    json_data = json.dumps(status) + "\n"
                    self.client_socket.sendall(json_data.encode('utf-8'))
                except Exception as e:
                    print(f"[UnityKeyframeReceiver] Status send error: {e}")
    
    def get_request(self, block: bool = False, timeout: float = None) -> Optional[InbetweenRequest]:
        """
        Get the next request from the queue.
        
        Args:
            block: If True, wait for a request. If False, return None if none available.
            timeout: Maximum time to wait (only used if block=True)
            
        Returns:
            InbetweenRequest or None
        """
        try:
            return self.request_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def has_request(self) -> bool:
        """Check if there's a request available"""
        return not self.request_queue.empty()
    
    def close(self):
        """Stop the server and close all connections"""
        self.is_running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
                
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        print("[UnityKeyframeReceiver] Server closed")


# Example usage
if __name__ == '__main__':
    def on_request(request):
        print(f"Callback received: {request}")
        print(f"  Start transl: {request.start_smpl['transl'] if request.start_smpl else 'None'}")
        print(f"  End transl: {request.end_smpl['transl'] if request.end_smpl else 'None'}")
    
    receiver = UnityKeyframeReceiver(on_request_received=on_request)
    
    print("Waiting for in-betweening requests... (Ctrl+C to exit)")
    try:
        while True:
            request = receiver.get_request(block=True, timeout=1.0)
            if request:
                print(f"Got request from queue: {request}")
    except KeyboardInterrupt:
        print("\nShutting down...")
        receiver.close()
