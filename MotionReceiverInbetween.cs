using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.IO;
using System;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Extended frame data for in-betweening support
/// </summary>
[System.Serializable]
public class ExtendedFrameData
{
    public string type;          // "frame" or "inbetween_frame"
    public float[] root_pos;
    public List<JointData> joints;
    public int frame_index;
    public int total_frames;
}

/// <summary>
/// Enhanced MotionReceiver with support for motion in-betweening.
/// Supports both regular streaming and buffered in-betweening playback with incremental blending.
/// </summary>
public class MotionReceiverInbetween : MonoBehaviour
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
    
    [Header("In-betweening Settings")]
    [Tooltip("How to blend in-betweening frames")]
    public InbetweenBlendMode blendMode = InbetweenBlendMode.Incremental;
    
    [Tooltip("Playback speed for in-betweening (1.0 = 30fps)")]
    public float inbetweenPlaybackSpeed = 1.0f;
    
    [Tooltip("Blend weight for incremental blending (0-1, higher = faster blend)")]
    [Range(0.1f, 1.0f)]
    public float blendWeight = 0.5f;
    
    [Header("Recording (Optional)")]
    public MotionRecorder recorder;
    
    [Header("Debug")]
    public bool debugMode = false;
    
    [Header("Status")]
    [SerializeField] private PlaybackMode currentMode = PlaybackMode.Streaming;
    [SerializeField] private int inbetweenBufferedFrames = 0;
    [SerializeField] private int inbetweenCurrentFrame = 0;
    [SerializeField] private int inbetweenTotalFrames = 0;
    
    // Events
    public event Action OnInbetweenStarted;
    public event Action<float> OnInbetweenProgress;  // progress 0-1
    public event Action OnInbetweenComplete;
    
    // Root Rotation Correction
    private float rootCorrectionX = -90f;
    private float rootCorrectionY = 0f;
    private float rootCorrectionZ = 0f;
    
    private TcpClient client;
    private NetworkStream stream;
    private StreamReader reader;
    
    // Background thread for network reading
    private Thread networkThread;
    private volatile bool isRunning = false;
    
    // Thread-safe frame storage
    private volatile string latestFrameJson = null;
    private readonly object frameLock = new object();
    
    // In-betweening buffer (for incremental blending)
    private Queue<ExtendedFrameData> inbetweenBuffer = new Queue<ExtendedFrameData>();
    private readonly object bufferLock = new object();
    
    // Blending state
    private ExtendedFrameData currentInbetweenFrame = null;
    private ExtendedFrameData targetInbetweenFrame = null;
    private float blendTimer = 0f;
    private float frameInterval = 1f / 30f;  // 30fps default
    
    // Stats
    private int framesReceived = 0;
    private int framesProcessed = 0;

    // Bone Mapping for SMPL joints to Unity Humanoid
    private Dictionary<string, HumanBodyBones> boneMap;
    
    // T-pose storage for SMPL avatar
    private Dictionary<HumanBodyBones, Quaternion> smplTPose;
    
    // HumanPose handlers for muscle-space retargeting
    private HumanPoseHandler smplPoseHandler;
    private HumanPoseHandler standardPoseHandler;
    private HumanPose humanPose;
    
    public enum PlaybackMode
    {
        Streaming,          // Normal continuous streaming
        InbetweenBuffering, // Receiving in-betweening frames
        InbetweenPlaying    // Playing back in-betweening frames
    }
    
    public enum InbetweenBlendMode
    {
        Immediate,      // Apply frames immediately as they arrive
        Incremental,    // Blend smoothly between frames
        BufferAndPlay   // Buffer all frames, then play back (not recommended)
    }
    
    public PlaybackMode Mode => currentMode;
    public float InbetweenProgress => inbetweenTotalFrames > 0 ? 
        (float)inbetweenCurrentFrame / inbetweenTotalFrames : 0f;

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
            Debug.Log("[MotionReceiverInbetween] Created HumanPoseHandler for SMPL avatar");
        }
        
        if (standardAnimator != null && standardAnimator.avatar != null && standardAnimator.avatar.isHuman)
        {
            standardPoseHandler = new HumanPoseHandler(standardAnimator.avatar, standardAnimator.transform);
            Debug.Log("[MotionReceiverInbetween] Created HumanPoseHandler for Standard avatar");
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
        
        Debug.Log($"[MotionReceiverInbetween] Captured SMPL T-pose: {smplTPose.Count} bones");
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
            Debug.Log("[MotionReceiverInbetween] Connected to DART Streamer!");
            
            // Start background network reading thread
            isRunning = true;
            networkThread = new Thread(NetworkReadLoop);
            networkThread.IsBackground = true;
            networkThread.Start();
            Debug.Log("[MotionReceiverInbetween] Started background network thread");
            
        } catch (Exception e) {
            Debug.LogError($"Connect Error: {e.Message}");
        }
    }
    
    /// <summary>
    /// Background thread that continuously reads from network.
    /// Routes frames to appropriate handlers based on type.
    /// </summary>
    void NetworkReadLoop()
    {
        while (isRunning && client != null && client.Connected)
        {
            try
            {
                string line = reader.ReadLine();
                
                if (!string.IsNullOrEmpty(line))
                {
                    // Try to parse as extended frame data first
                    var extFrame = JsonUtility.FromJson<ExtendedFrameData>(line);
                    
                    // Debug: Log what type was parsed
                    if (debugMode)
                    {
                        Debug.Log($"[MotionReceiverInbetween] Received frame: type='{extFrame?.type}', " +
                            $"frame_index={extFrame?.frame_index}, total_frames={extFrame?.total_frames}");
                    }
                    
                    if (extFrame != null && !string.IsNullOrEmpty(extFrame.type) && extFrame.type == "inbetween_frame")
                    {
                        // In-betweening frame - add to buffer
                        if (debugMode)
                        {
                            Debug.Log($"[MotionReceiverInbetween] Buffering inbetween frame {extFrame.frame_index}/{extFrame.total_frames}");
                        }
                        
                        lock (bufferLock)
                        {
                            inbetweenBuffer.Enqueue(extFrame);
                            inbetweenBufferedFrames = inbetweenBuffer.Count;
                            
                            if (extFrame.frame_index == 0)
                            {
                                // First frame - start in-betweening
                                inbetweenTotalFrames = extFrame.total_frames;
                                inbetweenCurrentFrame = 0;
                                currentMode = PlaybackMode.InbetweenBuffering;
                                Debug.Log($"[MotionReceiverInbetween] Started in-betweening mode, expecting {extFrame.total_frames} frames");
                            }
                        }
                        framesReceived++;
                    }
                    else
                    {
                        // Regular frame
                        lock (frameLock)
                        {
                            latestFrameJson = line;
                            framesReceived++;
                        }
                    }
                }
            }
            catch (IOException)
            {
                break;
            }
            catch (Exception e)
            {
                if (isRunning)
                {
                    Debug.LogWarning($"[MotionReceiverInbetween] Network read error: {e.Message}");
                }
                break;
            }
        }
        
        Debug.Log("[MotionReceiverInbetween] Network thread exited");
    }

    void Update()
    {
        switch (currentMode)
        {
            case PlaybackMode.Streaming:
                ProcessStreamingFrame();
                break;
                
            case PlaybackMode.InbetweenBuffering:
                // Incremental blending - start playing as soon as we have frames
                if (blendMode == InbetweenBlendMode.Incremental || 
                    blendMode == InbetweenBlendMode.Immediate)
                {
                    ProcessInbetweenFrames();
                }
                // Check if all frames received
                lock (bufferLock)
                {
                    if (inbetweenBufferedFrames >= inbetweenTotalFrames && inbetweenTotalFrames > 0)
                    {
                        if (blendMode == InbetweenBlendMode.BufferAndPlay)
                        {
                            currentMode = PlaybackMode.InbetweenPlaying;
                            OnInbetweenStarted?.Invoke();
                        }
                    }
                }
                break;
                
            case PlaybackMode.InbetweenPlaying:
                ProcessInbetweenFrames();
                break;
        }
    }
    
    void ProcessStreamingFrame()
    {
        // Get latest frame from background thread (non-blocking)
        string frameJson = null;
        lock (frameLock)
        {
            frameJson = latestFrameJson;
            latestFrameJson = null;
        }
        
        if (frameJson != null)
        {
            if (debugMode) Debug.Log($"[DART] Processing streaming frame");
            
            if (recorder != null)
            {
                recorder.RecordFrame(frameJson);
            }
            
            FrameData data = JsonUtility.FromJson<FrameData>(frameJson);
            if (data != null)
            {
                ApplyFrame(data);
            }
            framesProcessed++;
        }
    }
    
    void ProcessInbetweenFrames()
    {
        blendTimer += Time.deltaTime * inbetweenPlaybackSpeed;
        
        // Get next frame from buffer if needed
        while (blendTimer >= frameInterval)
        {
            blendTimer -= frameInterval;
            
            ExtendedFrameData nextFrame = null;
            lock (bufferLock)
            {
                if (inbetweenBuffer.Count > 0)
                {
                    nextFrame = inbetweenBuffer.Dequeue();
                    inbetweenBufferedFrames = inbetweenBuffer.Count;
                    inbetweenCurrentFrame = nextFrame.frame_index;
                }
            }
            
            if (nextFrame != null)
            {
                currentInbetweenFrame = targetInbetweenFrame;
                targetInbetweenFrame = nextFrame;
                
                OnInbetweenProgress?.Invoke(InbetweenProgress);
                
                // Check if complete
                if (nextFrame.frame_index >= nextFrame.total_frames - 1)
                {
                    // Apply final frame exactly
                    ApplyExtendedFrame(nextFrame);
                    
                    // Return to streaming mode
                    currentMode = PlaybackMode.Streaming;
                    OnInbetweenComplete?.Invoke();
                    
                    if (debugMode) Debug.Log("[MotionReceiverInbetween] In-betweening complete");
                    return;
                }
            }
            else if (currentMode == PlaybackMode.InbetweenPlaying)
            {
                // Buffer empty but still in playing mode - wait
                break;
            }
        }
        
        // Apply blended frame
        if (targetInbetweenFrame != null)
        {
            if (blendMode == InbetweenBlendMode.Immediate || currentInbetweenFrame == null)
            {
                ApplyExtendedFrame(targetInbetweenFrame);
            }
            else
            {
                // Incremental blend between current and target
                float t = Mathf.Clamp01(blendTimer / frameInterval);
                t = Mathf.Lerp(t, 1f, blendWeight);
                ApplyBlendedFrame(currentInbetweenFrame, targetInbetweenFrame, t);
            }
        }
    }
    
    void ApplyExtendedFrame(ExtendedFrameData extFrame)
    {
        // Convert to FrameData and apply
        FrameData data = new FrameData
        {
            root_pos = extFrame.root_pos,
            joints = extFrame.joints
        };
        ApplyFrame(data);
    }
    
    void ApplyBlendedFrame(ExtendedFrameData from, ExtendedFrameData to, float t)
    {
        if (from == null || to == null)
        {
            ApplyExtendedFrame(to ?? from);
            return;
        }
        
        // Blend root position
        Vector3 fromPos = new Vector3(from.root_pos[0], from.root_pos[1], from.root_pos[2]);
        Vector3 toPos = new Vector3(to.root_pos[0], to.root_pos[1], to.root_pos[2]);
        Vector3 blendedPos = Vector3.Lerp(fromPos, toPos, t);
        
        // Convert coordinate system
        Vector3 smplPos = new Vector3(blendedPos.x, blendedPos.z, -blendedPos.y);
        
        if (smplAnimator != null)
        {
            smplAnimator.transform.position = smplPos * smplPositionScale;
        }
        
        // Blend joint rotations
        Quaternion rootCorrection = Quaternion.Euler(rootCorrectionX, rootCorrectionY, rootCorrectionZ);
        
        // Create lookup for from joints
        Dictionary<string, Quaternion> fromJoints = new Dictionary<string, Quaternion>();
        if (from.joints != null)
        {
            foreach (var joint in from.joints)
            {
                if (joint.rot != null && joint.rot.Length == 4)
                {
                    Quaternion rot = new Quaternion(joint.rot[0], joint.rot[1], joint.rot[2], joint.rot[3]);
                    fromJoints[joint.name] = rot;
                }
            }
        }
        
        // Apply blended rotations
        if (to.joints != null)
        {
            foreach (var joint in to.joints)
            {
                if (!boneMap.ContainsKey(joint.name)) continue;
                if (joint.rot == null || joint.rot.Length != 4) continue;
                
                HumanBodyBones boneType = boneMap[joint.name];
                
                if (smplAnimator == null) continue;
                
                Transform bone = smplAnimator.GetBoneTransform(boneType);
                if (bone == null) continue;
                
                Quaternion toRot = new Quaternion(joint.rot[0], joint.rot[1], joint.rot[2], joint.rot[3]);
                toRot = ConvertRotationFromSMPL(toRot);
                
                Quaternion fromRot;
                if (fromJoints.TryGetValue(joint.name, out Quaternion rawFromRot))
                {
                    fromRot = ConvertRotationFromSMPL(rawFromRot);
                }
                else
                {
                    fromRot = toRot;
                }
                
                // Slerp between rotations
                Quaternion blendedRot = Quaternion.Slerp(fromRot, toRot, t);
                
                // Apply with T-pose offset
                Quaternion finalRot = smplTPose.ContainsKey(boneType) 
                    ? smplTPose[boneType] * blendedRot 
                    : blendedRot;
                
                if (boneType == HumanBodyBones.Hips)
                {
                    finalRot = rootCorrection * finalRot;
                }
                
                bone.localRotation = finalRot;
            }
        }
        
        // Copy pose from SMPL to Standard avatar via HumanPose
        if (smplPoseHandler != null && standardPoseHandler != null)
        {
            smplPoseHandler.GetHumanPose(ref humanPose);
            standardPoseHandler.SetHumanPose(ref humanPose);
            
            if (standardAnimator != null)
            {
                standardAnimator.transform.position = smplPos * standardPositionScale + standardPositionOffset;
            }
        }
    }

    void ApplyFrame(FrameData data)
    {
        if (data == null) return;

        // Process root position
        Vector3 smplPos = Vector3.zero;
        if (data.root_pos != null && data.root_pos.Length == 3)
        {
            smplPos = new Vector3(data.root_pos[0], data.root_pos[1], data.root_pos[2]);
            // Convert from SMPL coordinate system (Z-up) to Unity (Y-up)
            smplPos = new Vector3(smplPos.x, smplPos.z, -smplPos.y);
        }
        
        if (smplAnimator != null)
        {
            smplAnimator.transform.position = smplPos * smplPositionScale;
        }

        // Apply Joint Rotations
        Quaternion rootCorrection = Quaternion.Euler(rootCorrectionX, rootCorrectionY, rootCorrectionZ);
        
        if (data.joints != null)
        {
            foreach (var joint in data.joints)
            {
                if (!boneMap.ContainsKey(joint.name)) continue;
                
                HumanBodyBones boneType = boneMap[joint.name];
                
                if (smplAnimator == null) continue;
                
                Transform bone = smplAnimator.GetBoneTransform(boneType);
                if (bone == null) continue;
                
                if (joint.rot == null || joint.rot.Length != 4) continue;
                
                Quaternion smplRot = new Quaternion(joint.rot[0], joint.rot[1], joint.rot[2], joint.rot[3]);
                smplRot = ConvertRotationFromSMPL(smplRot);
                
                Quaternion finalRot = smplTPose.ContainsKey(boneType) 
                    ? smplTPose[boneType] * smplRot 
                    : smplRot;
                
                if (boneType == HumanBodyBones.Hips)
                {
                    finalRot = rootCorrection * finalRot;
                }
                
                bone.localRotation = finalRot;
            }
        }
        
        // Copy pose from SMPL to Standard avatar via HumanPose
        if (smplPoseHandler != null && standardPoseHandler != null)
        {
            smplPoseHandler.GetHumanPose(ref humanPose);
            standardPoseHandler.SetHumanPose(ref humanPose);
            
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
    
    /// <summary>
    /// Manually trigger return to streaming mode (e.g., if in-betweening is cancelled)
    /// </summary>
    public void ReturnToStreaming()
    {
        lock (bufferLock)
        {
            inbetweenBuffer.Clear();
            inbetweenBufferedFrames = 0;
        }
        currentMode = PlaybackMode.Streaming;
        currentInbetweenFrame = null;
        targetInbetweenFrame = null;
    }

    void OnDestroy()
    {
        isRunning = false;
        
        if (networkThread != null && networkThread.IsAlive)
        {
            networkThread.Join(100);
        }
        
        if (client != null) client.Close();
    }
    
    void OnGUI()
    {
        if (!debugMode) return;
        
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 12;
        
        GUI.Label(new Rect(10, 50, 400, 20), 
            $"Mode: {currentMode}, Received: {framesReceived}, Processed: {framesProcessed}", style);
        
        if (currentMode != PlaybackMode.Streaming)
        {
            GUI.Label(new Rect(10, 70, 400, 20),
                $"In-betweening: {inbetweenCurrentFrame}/{inbetweenTotalFrames}, Buffer: {inbetweenBufferedFrames}", style);
        }
    }
}
