using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.IO;
using System;

[System.Serializable]
public class JointData
{
    public string name;
    public float[] rot; // [x, y, z, w]
}

[System.Serializable]
public class FrameData
{
    public float[] root_pos;
    public List<JointData> joints;
}

/// <summary>
/// Receives animation data from DART streamer and applies to Unity Humanoid avatars.
/// Uses HumanPoseHandler for proper muscle-space retargeting between different avatar skeletons.
/// </summary>
public class MotionReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    public string host = "127.0.0.1";
    public int port = 8080;
    
    [Header("SMPL Avatar (Reference)")]
    [Tooltip("The SMPL for Unity avatar - receives direct SMPL rotations")]
    public Animator smplAnimator;
    public float smplPositionScale = 1.0f;
    
    [Header("Standard Avatar (Retargeted)")]
    [Tooltip("Standard Unity humanoid avatar - pose is copied from SMPL via HumanPose")]
    public Animator standardAnimator;
    public float standardPositionScale = 1.0f;
    [Tooltip("Position offset for side-by-side comparison")]
    public Vector3 standardPositionOffset = new Vector3(1.5f, 0, 0);
    
    [Header("Coordinate Conversion")]
    [Tooltip("Rotation conversion index. 6 = (-x, y, -z, w) works for DART")]
    public int rotationConversionIndex = 6;
    
    [Header("Root Rotation Correction")]
    [Tooltip("Rotation around X axis for root. -90 works for DART")]
    public float rootCorrectionX = -90f;
    public float rootCorrectionY = 0f;
    public float rootCorrectionZ = 0f;
    
    [Header("Recording (Optional)")]
    public MotionRecorder recorder;
    
    [Header("Debug")]
    public bool debugMode = false;
    
    private TcpClient client;
    private NetworkStream stream;
    private StreamReader reader;

    // Bone Mapping for SMPL joints to Unity Humanoid
    private Dictionary<string, HumanBodyBones> boneMap;
    
    // T-pose storage for SMPL avatar
    private Dictionary<HumanBodyBones, Quaternion> smplTPose;
    
    // HumanPose handlers for muscle-space retargeting
    private HumanPoseHandler smplPoseHandler;
    private HumanPoseHandler standardPoseHandler;
    private HumanPose humanPose;

    void Start()
    {
        InitializeBoneMap();
        CaptureSmplTPose();
        InitializeHumanPoseHandlers();
        Connect();
    }
    
    void InitializeHumanPoseHandlers()
    {
        if (smplAnimator != null && smplAnimator.avatar != null && smplAnimator.avatar.isHuman)
        {
            smplPoseHandler = new HumanPoseHandler(smplAnimator.avatar, smplAnimator.transform);
            Debug.Log("[MotionReceiver] Created HumanPoseHandler for SMPL avatar");
        }
        
        if (standardAnimator != null && standardAnimator.avatar != null && standardAnimator.avatar.isHuman)
        {
            standardPoseHandler = new HumanPoseHandler(standardAnimator.avatar, standardAnimator.transform);
            Debug.Log("[MotionReceiver] Created HumanPoseHandler for Standard avatar");
        }
        
        humanPose = new HumanPose();
    }

    void InitializeBoneMap()
    {
        boneMap = new Dictionary<string, HumanBodyBones>
        {
            ["Hips"] = HumanBodyBones.Hips,
            ["Spine"] = HumanBodyBones.Spine,
            ["Chest"] = HumanBodyBones.Chest,
            ["UpperChest"] = HumanBodyBones.UpperChest,
            ["Neck"] = HumanBodyBones.Neck,
            ["Head"] = HumanBodyBones.Head,
            
            ["LeftUpperLeg"] = HumanBodyBones.LeftUpperLeg,
            ["LeftLowerLeg"] = HumanBodyBones.LeftLowerLeg,
            ["LeftFoot"] = HumanBodyBones.LeftFoot,
            ["LeftToes"] = HumanBodyBones.LeftToes,
            
            ["RightUpperLeg"] = HumanBodyBones.RightUpperLeg,
            ["RightLowerLeg"] = HumanBodyBones.RightLowerLeg,
            ["RightFoot"] = HumanBodyBones.RightFoot,
            ["RightToes"] = HumanBodyBones.RightToes,
            
            ["LeftShoulder"] = HumanBodyBones.LeftShoulder,
            ["LeftUpperArm"] = HumanBodyBones.LeftUpperArm,
            ["LeftLowerArm"] = HumanBodyBones.LeftLowerArm,
            ["LeftHand"] = HumanBodyBones.LeftHand,
            
            ["RightShoulder"] = HumanBodyBones.RightShoulder,
            ["RightUpperArm"] = HumanBodyBones.RightUpperArm,
            ["RightLowerArm"] = HumanBodyBones.RightLowerArm,
            ["RightHand"] = HumanBodyBones.RightHand
        };
    }
    
    void CaptureSmplTPose()
    {
        smplTPose = new Dictionary<HumanBodyBones, Quaternion>();
        
        if (smplAnimator == null) return;
        
        foreach (var kvp in boneMap)
        {
            Transform bone = smplAnimator.GetBoneTransform(kvp.Value);
            if (bone != null)
            {
                smplTPose[kvp.Value] = bone.localRotation;
            }
        }
        
        Debug.Log($"[MotionReceiver] Captured SMPL T-pose: {smplTPose.Count} bones");
    }

    void Connect()
    {
        try {
            client = new TcpClient();
            client.BeginConnect(host, port, OnConnect, null);
        } catch (Exception e) {
            Debug.LogError($"Connection failed: {e.Message}");
        }
    }

    void OnConnect(IAsyncResult ar)
    {
        try {
            client.EndConnect(ar);
            stream = client.GetStream();
            reader = new StreamReader(stream, Encoding.UTF8);
            Debug.Log("Connected to DART Streamer!");
        } catch (Exception e) {
            Debug.LogError($"Connect Error: {e.Message}");
        }
    }

    void Update()
    {
        if (client == null || !client.Connected) return;

        if (stream.DataAvailable)
        {
            try {
                string lastLine = null;
                while (stream.DataAvailable) {
                    string line = reader.ReadLine();
                    if (!string.IsNullOrEmpty(line)) lastLine = line;
                }

                if (lastLine != null) {
                    if (debugMode) Debug.Log($"[DART] Received JSON: {lastLine}");
                    
                    if (recorder != null)
                    {
                        recorder.RecordFrame(lastLine);
                    }
                    
                    ProcessFrame(lastLine);
                }
            } catch (Exception e) {
                Debug.LogWarning($"Read Error: {e.Message}");
            }
        }
    }

    void ProcessFrame(string json)
    {
        FrameData data = JsonUtility.FromJson<FrameData>(json);
        if (data == null) return;

        // Process root position
        Vector3 smplPos = Vector3.zero;
        if (data.root_pos != null && data.root_pos.Length == 3)
        {
            smplPos = new Vector3(data.root_pos[0], data.root_pos[1], data.root_pos[2]);
            // Convert from SMPL coordinate system (Z-up) to Unity (Y-up)
            smplPos = new Vector3(smplPos.x, smplPos.z, -smplPos.y);
        }
        
        // Apply position to SMPL avatar
        if (smplAnimator != null)
        {
            smplAnimator.transform.position = smplPos * smplPositionScale;
        }

        // Apply Joint Rotations to SMPL avatar
        Quaternion rootCorrection = Quaternion.Euler(rootCorrectionX, rootCorrectionY, rootCorrectionZ);
        
        foreach (var joint in data.joints)
        {
            if (!boneMap.ContainsKey(joint.name)) continue;
            
            HumanBodyBones boneType = boneMap[joint.name];
            
            if (smplAnimator == null) continue;
            
            Transform bone = smplAnimator.GetBoneTransform(boneType);
            if (bone == null) continue;
            
            // Get and convert rotation
            Quaternion smplRot = new Quaternion(joint.rot[0], joint.rot[1], joint.rot[2], joint.rot[3]);
            smplRot = ConvertRotationFromSMPL(smplRot);
            
            // Apply with T-pose offset
            Quaternion finalRot = smplTPose.ContainsKey(boneType) 
                ? smplTPose[boneType] * smplRot 
                : smplRot;
            
            // Root correction for Hips
            if (boneType == HumanBodyBones.Hips)
            {
                finalRot = rootCorrection * finalRot;
            }
            
            bone.localRotation = finalRot;
        }
        
        // Copy pose from SMPL to Standard avatar via HumanPose (muscle space)
        if (smplPoseHandler != null && standardPoseHandler != null)
        {
            smplPoseHandler.GetHumanPose(ref humanPose);
            standardPoseHandler.SetHumanPose(ref humanPose);
            
            // Apply position offset for comparison view
            if (standardAnimator != null)
            {
                standardAnimator.transform.position = smplPos * standardPositionScale + standardPositionOffset;
            }
        }
    }
    
    Quaternion ConvertRotationFromSMPL(Quaternion q)
    {
        // Index 6: (-x, y, -z, w) - works for DART
        return new Quaternion(-q.x, q.y, -q.z, q.w);
    }

    void OnDestroy()
    {
        if (client != null) client.Close();
    }
}
