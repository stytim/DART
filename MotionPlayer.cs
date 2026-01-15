using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

/// <summary>
/// Plays back recorded animation data from a JSON file.
/// Uses HumanPoseHandler for proper muscle-space retargeting between different avatar skeletons.
/// </summary>
public class MotionPlayer : MonoBehaviour
{
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
    
    [Header("Playback Settings")]
    [Tooltip("Path to the recording file (absolute or relative to persistentDataPath)")]
    public string recordingFilePath = "";
    
    [Tooltip("Playback speed multiplier")]
    public float playbackSpeed = 1.0f;
    
    [Tooltip("Loop the animation")]
    public bool loop = true;
    
    [Header("Root Rotation Correction")]
    [Tooltip("Rotation around X axis for root. -90 works for DART")]
    public float rootCorrectionX = -90f;
    public float rootCorrectionY = 0f;
    public float rootCorrectionZ = 0f;
    
    [Header("Controls")]
    public bool isPlaying = false;
    
    [Header("Status (Read Only)")]
    [SerializeField] private int currentFrameIndex = 0;
    [SerializeField] private int totalFrames = 0;
    [SerializeField] private float currentTime = 0f;
    
    [Header("Debug")]
    public bool debugMode = false;
    
    private List<RecordedFrame> frames = new List<RecordedFrame>();
    private float playbackStartTime = 0f;
    private Dictionary<string, HumanBodyBones> boneMap;
    
    // T-pose storage for SMPL avatar
    private Dictionary<HumanBodyBones, Quaternion> smplTPose;
    
    // HumanPose handlers for muscle-space retargeting
    private HumanPoseHandler smplPoseHandler;
    private HumanPoseHandler standardPoseHandler;
    private HumanPose humanPose;
    
    [System.Serializable]
    public class RecordedFrame
    {
        public float timestamp;
        public string rawJson;
    }
    
    [System.Serializable]
    public class RecordingData
    {
        public List<RecordedFrame> frames;
    }
    
    [System.Serializable]
    public class JointData
    {
        public string name;
        public float[] rot;
    }
    
    [System.Serializable]
    public class FrameData
    {
        public float[] root_pos;
        public List<JointData> joints;
    }
    
    void Start()
    {
        InitializeBoneMap();
        CaptureSmplTPose();
        InitializeHumanPoseHandlers();
        LoadRecording();
        Play();
    }
    
    void InitializeHumanPoseHandlers()
    {
        if (smplAnimator != null && smplAnimator.avatar != null && smplAnimator.avatar.isHuman)
        {
            smplPoseHandler = new HumanPoseHandler(smplAnimator.avatar, smplAnimator.transform);
            Debug.Log("[MotionPlayer] Created HumanPoseHandler for SMPL avatar");
        }
        
        if (standardAnimator != null && standardAnimator.avatar != null && standardAnimator.avatar.isHuman)
        {
            standardPoseHandler = new HumanPoseHandler(standardAnimator.avatar, standardAnimator.transform);
            Debug.Log("[MotionPlayer] Created HumanPoseHandler for Standard avatar");
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
        
        Debug.Log($"[MotionPlayer] Captured SMPL T-pose: {smplTPose.Count} bones");
    }
    
    void Update()
    {
        if (!isPlaying || frames.Count == 0 || smplAnimator == null) return;
        
        currentTime = (Time.time - playbackStartTime) * playbackSpeed;
        
        int frameIndex = FindFrameForTime(currentTime);
        
        if (frameIndex >= frames.Count)
        {
            if (loop)
            {
                playbackStartTime = Time.time;
                currentTime = 0f;
                currentFrameIndex = 0;
            }
            else
            {
                isPlaying = false;
                Debug.Log("[MotionPlayer] Playback finished.");
            }
            return;
        }
        
        if (frameIndex != currentFrameIndex)
        {
            currentFrameIndex = frameIndex;
            ApplyFrame(frames[currentFrameIndex].rawJson);
        }
    }
    
    int FindFrameForTime(float time)
    {
        for (int i = 0; i < frames.Count; i++)
        {
            if (frames[i].timestamp > time)
            {
                return Mathf.Max(0, i - 1);
            }
        }
        return frames.Count;
    }
    
    /// <summary>
    /// Load a recording from file
    /// </summary>
    public void LoadRecording()
    {
        LoadRecording(recordingFilePath);
    }
    
    public void LoadRecording(string path)
    {
        if (string.IsNullOrEmpty(path))
        {
            Debug.LogWarning("[MotionPlayer] No recording file path specified.");
            return;
        }
        
        string fullPath = path;
        if (!Path.IsPathRooted(path))
        {
            fullPath = Path.Combine(Application.persistentDataPath, path);
        }
        
        if (!File.Exists(fullPath))
        {
            Debug.LogError($"[MotionPlayer] Recording file not found: {fullPath}");
            return;
        }
        
        try
        {
            string json = File.ReadAllText(fullPath);
            RecordingData data = JsonUtility.FromJson<RecordingData>(json);
            
            if (data != null && data.frames != null)
            {
                frames = data.frames;
                totalFrames = frames.Count;
                currentFrameIndex = 0;
                Debug.Log($"[MotionPlayer] Loaded {frames.Count} frames from {fullPath}");
            }
            else
            {
                Debug.LogError("[MotionPlayer] Failed to parse recording file.");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[MotionPlayer] Error loading recording: {e.Message}");
        }
    }
    
    /// <summary>
    /// Start playback
    /// </summary>
    public void Play()
    {
        if (frames.Count == 0)
        {
            Debug.LogWarning("[MotionPlayer] No frames loaded.");
            return;
        }
        
        playbackStartTime = Time.time;
        currentFrameIndex = 0;
        currentTime = 0f;
        isPlaying = true;
        
        Debug.Log("[MotionPlayer] Playback started.");
    }
    
    /// <summary>
    /// Stop playback
    /// </summary>
    public void Stop()
    {
        isPlaying = false;
        currentFrameIndex = 0;
        currentTime = 0f;
        Debug.Log("[MotionPlayer] Playback stopped.");
    }
    
    /// <summary>
    /// Pause playback
    /// </summary>
    public void Pause()
    {
        isPlaying = false;
        Debug.Log("[MotionPlayer] Playback paused.");
    }
    
    /// <summary>
    /// Resume playback
    /// </summary>
    public void Resume()
    {
        if (frames.Count == 0) return;
        
        playbackStartTime = Time.time - (currentTime / playbackSpeed);
        isPlaying = true;
        
        Debug.Log("[MotionPlayer] Playback resumed.");
    }
    
    void ApplyFrame(string json)
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
            
            // Get and convert rotation - Index 6: (-x, y, -z, w) works for DART
            Quaternion smplRot = new Quaternion(joint.rot[0], joint.rot[1], joint.rot[2], joint.rot[3]);
            smplRot = new Quaternion(-smplRot.x, smplRot.y, -smplRot.z, smplRot.w);
            
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
}
