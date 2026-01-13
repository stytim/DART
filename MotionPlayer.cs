using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

/// <summary>
/// Plays back recorded animation data from a JSON file.
/// Works with the same data format as MotionReceiver.
/// </summary>
public class MotionPlayer : MonoBehaviour
{
    [Header("Target Settings")]
    [Tooltip("Assign the Animator component of the character you want to animate.")]
    public Animator targetAnimator;
    public float positionScale = 1.0f;
    
    [Header("Playback Settings")]
    [Tooltip("Path to the recording file (absolute or relative to persistentDataPath)")]
    public string recordingFilePath = "";
    
    [Tooltip("Playback speed multiplier")]
    public float playbackSpeed = 1.0f;
    
    [Tooltip("Loop the animation")]
    public bool loop = true;
    
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
    
    [System.Serializable]
    public class RecordedFrame
    {
        public float timestamp;
        public string rawJson;
    }
    
    [System.Serializable]
    public class RecordingData
    {
        public string createdAt;
        public float totalDuration;
        public int totalFrames;
        public float averageFps;
        public List<RecordedFrame> frames;
    }
    
    // Frame data classes (same as MotionReceiver)
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
        
        if (targetAnimator == null)
        {
            targetAnimator = GetComponent<Animator>();
        }
        
        if (targetAnimator == null)
        {
            Debug.LogError("MotionPlayer: No Target Animator assigned!");
        }
    }
    
    void InitializeBoneMap()
    {
        boneMap = new Dictionary<string, HumanBodyBones>();
        boneMap["Hips"] = HumanBodyBones.Hips;
        boneMap["Spine"] = HumanBodyBones.Spine;
        boneMap["Chest"] = HumanBodyBones.Chest;
        boneMap["UpperChest"] = HumanBodyBones.UpperChest;
        boneMap["Neck"] = HumanBodyBones.Neck;
        boneMap["Head"] = HumanBodyBones.Head;
        
        boneMap["LeftUpperLeg"] = HumanBodyBones.LeftUpperLeg;
        boneMap["LeftLowerLeg"] = HumanBodyBones.LeftLowerLeg;
        boneMap["LeftFoot"] = HumanBodyBones.LeftFoot;
        boneMap["LeftToes"] = HumanBodyBones.LeftToes;
        
        boneMap["RightUpperLeg"] = HumanBodyBones.RightUpperLeg;
        boneMap["RightLowerLeg"] = HumanBodyBones.RightLowerLeg;
        boneMap["RightFoot"] = HumanBodyBones.RightFoot;
        boneMap["RightToes"] = HumanBodyBones.RightToes;
        
        boneMap["LeftShoulder"] = HumanBodyBones.LeftShoulder;
        boneMap["LeftUpperArm"] = HumanBodyBones.LeftUpperArm;
        boneMap["LeftLowerArm"] = HumanBodyBones.LeftLowerArm;
        boneMap["LeftHand"] = HumanBodyBones.LeftHand;
        
        boneMap["RightShoulder"] = HumanBodyBones.RightShoulder;
        boneMap["RightUpperArm"] = HumanBodyBones.RightUpperArm;
        boneMap["RightLowerArm"] = HumanBodyBones.RightLowerArm;
        boneMap["RightHand"] = HumanBodyBones.RightHand;
    }
    
    void Update()
    {
        if (!isPlaying || frames.Count == 0 || targetAnimator == null) return;
        
        currentTime = (Time.time - playbackStartTime) * playbackSpeed;
        
        // Find the appropriate frame for current time
        int frameIndex = FindFrameForTime(currentTime);
        
        if (frameIndex >= frames.Count)
        {
            if (loop)
            {
                // Reset playback
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
    
    private int FindFrameForTime(float time)
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
    
    /// <summary>
    /// Load a recording from file
    /// </summary>
    public void LoadRecording(string filePath)
    {
        string fullPath = filePath;
        
        // Check if relative path
        if (!File.Exists(fullPath))
        {
            fullPath = Path.Combine(Application.persistentDataPath, filePath);
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
            
            frames = data.frames;
            totalFrames = data.totalFrames;
            currentFrameIndex = 0;
            currentTime = 0f;
            
            Debug.Log($"[MotionPlayer] Loaded recording: {totalFrames} frames, {data.totalDuration:F2}s duration");
        }
        catch (Exception e)
        {
            Debug.LogError($"[MotionPlayer] Failed to load recording: {e.Message}");
        }
    }
    
    /// <summary>
    /// Start playback
    /// </summary>
    public void Play()
    {
        if (frames.Count == 0)
        {
            Debug.LogWarning("[MotionPlayer] No recording loaded. Call LoadRecording first.");
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
        
        // Adjust start time to maintain position
        playbackStartTime = Time.time - (currentTime / playbackSpeed);
        isPlaying = true;
        
        Debug.Log("[MotionPlayer] Playback resumed.");
    }
    
    private void ApplyFrame(string json)
    {
        FrameData data = JsonUtility.FromJson<FrameData>(json);
        if (data == null) return;
        
        // Apply Root Position
        if (data.root_pos != null && data.root_pos.Length == 3)
        {
            Vector3 pos = new Vector3(data.root_pos[0], data.root_pos[1], data.root_pos[2]);
            if (debugMode) Debug.Log($"[MotionPlayer] Root Pos: {pos * positionScale}");
            targetAnimator.transform.position = pos * positionScale;
        }
        
        // Apply Joint Rotations
        foreach (var joint in data.joints)
        {
            if (boneMap.ContainsKey(joint.name))
            {
                Transform bone = targetAnimator.GetBoneTransform(boneMap[joint.name]);
                if (bone != null)
                {
                    Quaternion rot = new Quaternion(joint.rot[0], joint.rot[1], joint.rot[2], joint.rot[3]);
                    bone.localRotation = rot;
                }
            }
        }
    }
}
