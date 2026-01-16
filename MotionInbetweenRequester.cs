using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.IO;
using System;
using System.Threading;

/// <summary>
/// Requests motion in-betweening from DART server.
/// Extracts keyframes from Unity Humanoid avatars and sends to DART for transition generation.
/// Uses HumanPoseHandler to convert from Unity's muscle space to rotation space.
/// </summary>
public class MotionInbetweenRequester : MonoBehaviour
{
    [Header("Network Settings")]
    [Tooltip("IP address of the DART server")]
    public string host = "127.0.0.1";
    
    [Tooltip("Port for keyframe receiver (default 8082)")]
    public int keyframePort = 8082;
    
    [Tooltip("Auto-connect on start")]
    public bool autoConnect = true;
    
    [Header("Avatar Reference")]
    [Tooltip("The SMPL avatar to extract poses from")]
    public Animator smplAnimator;
    
    [Header("In-betweening Settings")]
    [Tooltip("Default duration of generated transition in frames (30fps)")]
    public int defaultDurationFrames = 60;
    
    [Tooltip("Default text prompt for motion generation")]
    public string defaultPrompt = "walk forward";
    
    [Header("Status")]
    [SerializeField] private bool isConnected = false;
    [SerializeField] private InbetweenState currentState = InbetweenState.Idle;
    [SerializeField] private int totalFrames = 0;
    [SerializeField] private int receivedFrames = 0;
    
    // Events for external scripts to listen to
    public event Action OnGenerationStarted;
    public event Action<int, int> OnFrameReceived; // frameIndex, totalFrames
    public event Action<bool> OnGenerationComplete; // success
    
    private TcpClient client;
    private NetworkStream stream;
    private StreamReader reader;
    private Thread receiveThread;
    private volatile bool isRunning = false;
    
    // HumanPose extraction
    private HumanPoseHandler poseHandler;
    private HumanPose humanPose;
    
    // Joint mapping for SMPL (22 joints)
    private readonly string[] jointNames = {
        "Hips", "LeftUpperLeg", "RightUpperLeg", "Spine",
        "LeftLowerLeg", "RightLowerLeg", "Chest",
        "LeftFoot", "RightFoot", "UpperChest",
        "LeftToes", "RightToes", "Neck",
        "LeftShoulder", "RightShoulder", "Head",
        "LeftUpperArm", "RightUpperArm",
        "LeftLowerArm", "RightLowerArm",
        "LeftHand", "RightHand"
    };
    
    private readonly Dictionary<string, HumanBodyBones> boneMap = new Dictionary<string, HumanBodyBones>
    {
        {"Hips", HumanBodyBones.Hips},
        {"LeftUpperLeg", HumanBodyBones.LeftUpperLeg},
        {"RightUpperLeg", HumanBodyBones.RightUpperLeg},
        {"Spine", HumanBodyBones.Spine},
        {"LeftLowerLeg", HumanBodyBones.LeftLowerLeg},
        {"RightLowerLeg", HumanBodyBones.RightLowerLeg},
        {"Chest", HumanBodyBones.Chest},
        {"LeftFoot", HumanBodyBones.LeftFoot},
        {"RightFoot", HumanBodyBones.RightFoot},
        {"UpperChest", HumanBodyBones.UpperChest},
        {"LeftToes", HumanBodyBones.LeftToes},
        {"RightToes", HumanBodyBones.RightToes},
        {"Neck", HumanBodyBones.Neck},
        {"LeftShoulder", HumanBodyBones.LeftShoulder},
        {"RightShoulder", HumanBodyBones.RightShoulder},
        {"Head", HumanBodyBones.Head},
        {"LeftUpperArm", HumanBodyBones.LeftUpperArm},
        {"RightUpperArm", HumanBodyBones.RightUpperArm},
        {"LeftLowerArm", HumanBodyBones.LeftLowerArm},
        {"RightLowerArm", HumanBodyBones.RightLowerArm},
        {"LeftHand", HumanBodyBones.LeftHand},
        {"RightHand", HumanBodyBones.RightHand}
    };
    
    public enum InbetweenState
    {
        Idle,
        Requesting,
        Generating,
        Streaming,
        Complete,
        Error
    }
    
    public InbetweenState State => currentState;
    public float Progress => totalFrames > 0 ? (float)receivedFrames / totalFrames : 0f;
    
    void Start()
    {
        InitializeHumanPoseHandler();
        
        if (autoConnect)
        {
            Connect();
        }
    }
    
    void InitializeHumanPoseHandler()
    {
        if (smplAnimator != null && smplAnimator.avatar != null && smplAnimator.avatar.isHuman)
        {
            poseHandler = new HumanPoseHandler(smplAnimator.avatar, smplAnimator.transform);
            humanPose = new HumanPose();
            Debug.Log("[InbetweenRequester] Initialized HumanPoseHandler for pose extraction");
        }
        else
        {
            Debug.LogWarning("[InbetweenRequester] SMPL Animator not set or not humanoid!");
        }
    }
    
    /// <summary>
    /// Connect to the DART keyframe receiver
    /// </summary>
    public void Connect()
    {
        if (isConnected) return;
        
        try
        {
            client = new TcpClient();
            client.BeginConnect(host, keyframePort, OnConnected, null);
            Debug.Log($"[InbetweenRequester] Connecting to {host}:{keyframePort}...");
        }
        catch (Exception e)
        {
            Debug.LogError($"[InbetweenRequester] Connection failed: {e.Message}");
            currentState = InbetweenState.Error;
        }
    }
    
    void OnConnected(IAsyncResult ar)
    {
        try
        {
            client.EndConnect(ar);
            stream = client.GetStream();
            reader = new StreamReader(stream, Encoding.UTF8);
            isConnected = true;
            currentState = InbetweenState.Idle;
            
            // Start receive thread
            isRunning = true;
            receiveThread = new Thread(ReceiveLoop);
            receiveThread.IsBackground = true;
            receiveThread.Start();
            
            Debug.Log("[InbetweenRequester] Connected to DART keyframe receiver!");
        }
        catch (Exception e)
        {
            Debug.LogError($"[InbetweenRequester] Connection error: {e.Message}");
            isConnected = false;
            currentState = InbetweenState.Error;
        }
    }
    
    void ReceiveLoop()
    {
        while (isRunning && client != null && client.Connected)
        {
            try
            {
                string line = reader.ReadLine();
                if (!string.IsNullOrEmpty(line))
                {
                    // Parse on main thread
                    ProcessMessage(line);
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
                    Debug.LogWarning($"[InbetweenRequester] Receive error: {e.Message}");
                }
                break;
            }
        }
    }
    
    void ProcessMessage(string json)
    {
        try
        {
            var response = JsonUtility.FromJson<InbetweenResponse>(json);
            
            switch (response.type)
            {
                case "inbetween_ack":
                    Debug.Log($"[InbetweenRequester] Request acknowledged, processing {response.duration_frames} frames");
                    currentState = InbetweenState.Generating;
                    totalFrames = response.duration_frames;
                    receivedFrames = 0;
                    break;
                    
                case "inbetween_status":
                    Debug.Log($"[InbetweenRequester] Status: {response.status} - {response.message}");
                    if (response.status == "streaming")
                    {
                        currentState = InbetweenState.Streaming;
                        totalFrames = response.total_frames;
                    }
                    break;
                    
                case "inbetween_complete":
                    Debug.Log($"[InbetweenRequester] Complete! {response.total_frames} frames, " +
                              $"generation time: {response.generation_time_ms}ms");
                    currentState = InbetweenState.Complete;
                    OnGenerationComplete?.Invoke(response.success);
                    break;
                    
                case "inbetween_error":
                    Debug.LogError($"[InbetweenRequester] Error: {response.message}");
                    currentState = InbetweenState.Error;
                    OnGenerationComplete?.Invoke(false);
                    break;
                    
                case "pong":
                    Debug.Log("[InbetweenRequester] Pong received");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[InbetweenRequester] Failed to parse response: {e.Message}");
        }
    }
    
    /// <summary>
    /// Request in-betweening using current avatar pose as start and provided animation clip pose as end
    /// </summary>
    /// <param name="targetClip">Animation clip to extract end pose from</param>
    /// <param name="targetTime">Time in the clip to sample the end pose</param>
    /// <param name="prompt">Text description of the transition motion</param>
    /// <param name="durationFrames">Number of frames for the transition</param>
    public void RequestInbetween(AnimationClip targetClip, float targetTime, string prompt = null, int durationFrames = -1)
    {
        if (!isConnected)
        {
            Debug.LogError("[InbetweenRequester] Not connected!");
            return;
        }
        
        if (smplAnimator == null)
        {
            Debug.LogError("[InbetweenRequester] SMPL Animator not assigned!");
            return;
        }
        
        // Extract current pose as start frame
        var startFrame = ExtractCurrentPose();
        
        // Sample target clip for end frame
        var endFrame = ExtractPoseFromClip(targetClip, targetTime);
        
        // Send request
        SendInbetweenRequest(startFrame, endFrame, prompt ?? defaultPrompt, 
                            durationFrames > 0 ? durationFrames : defaultDurationFrames);
    }
    
    /// <summary>
    /// Request in-betweening between two specific poses
    /// </summary>
    public void RequestInbetween(KeyframeData startFrame, KeyframeData endFrame, 
                                  string prompt = null, int durationFrames = -1)
    {
        if (!isConnected)
        {
            Debug.LogError("[InbetweenRequester] Not connected!");
            return;
        }
        
        SendInbetweenRequest(startFrame, endFrame, prompt ?? defaultPrompt,
                            durationFrames > 0 ? durationFrames : defaultDurationFrames);
    }
    
    /// <summary>
    /// Request in-betweening from current pose to a target pose on another animator
    /// </summary>
    public void RequestInbetweenToTarget(Animator targetAnimator, string prompt = null, int durationFrames = -1)
    {
        if (!isConnected)
        {
            Debug.LogError("[InbetweenRequester] Not connected!");
            return;
        }
        
        var startFrame = ExtractCurrentPose();
        var endFrame = ExtractPoseFromAnimator(targetAnimator);
        
        SendInbetweenRequest(startFrame, endFrame, prompt ?? defaultPrompt,
                            durationFrames > 0 ? durationFrames : defaultDurationFrames);
    }
    
    void SendInbetweenRequest(KeyframeData startFrame, KeyframeData endFrame, string prompt, int durationFrames)
    {
        currentState = InbetweenState.Requesting;
        totalFrames = durationFrames;
        receivedFrames = 0;
        
        var request = new InbetweenRequest
        {
            type = "inbetween_request",
            start_frame = startFrame,
            end_frame = endFrame,
            prompt = prompt,
            duration_frames = durationFrames,
            mode = "fast"
        };
        
        string json = JsonUtility.ToJson(request) + "\n";
        byte[] data = Encoding.UTF8.GetBytes(json);
        
        try
        {
            stream.Write(data, 0, data.Length);
            Debug.Log($"[InbetweenRequester] Sent request: prompt='{prompt}', frames={durationFrames}");
            OnGenerationStarted?.Invoke();
        }
        catch (Exception e)
        {
            Debug.LogError($"[InbetweenRequester] Send error: {e.Message}");
            currentState = InbetweenState.Error;
        }
    }
    
    /// <summary>
    /// Extract current pose from the SMPL animator
    /// </summary>
    public KeyframeData ExtractCurrentPose()
    {
        if (smplAnimator == null) return new KeyframeData();
        
        var frame = new KeyframeData();
        
        // Get root position
        Transform hips = smplAnimator.GetBoneTransform(HumanBodyBones.Hips);
        if (hips != null)
        {
            Vector3 pos = hips.position;
            // Convert to SMPL coordinate system (same as MotionReceiver but reversed)
            // Unity: Y-up, Z-forward -> SMPL: Z-up, -Y-forward
            frame.root_pos = new float[] { -pos.x, -pos.z, pos.y };
        }
        
        // Get joint rotations
        frame.joints = new List<JointRotation>();
        
        foreach (var kvp in boneMap)
        {
            Transform bone = smplAnimator.GetBoneTransform(kvp.Value);
            if (bone != null)
            {
                Quaternion rot = bone.localRotation;
                
                // Convert from Unity to SMPL rotation
                // Reverse of MotionReceiver.ConvertRotationFromSMPL: (-x, y, -z, w) -> SMPL
                // So SMPL = (-x, -y, z, w) to reverse the conversion
                var jointRot = new JointRotation
                {
                    name = kvp.Key,
                    rot = new float[] { -rot.x, -rot.y, rot.z, rot.w }
                };
                frame.joints.Add(jointRot);
            }
        }
        
        return frame;
    }
    
    /// <summary>
    /// Extract pose from an animation clip at a specific time
    /// </summary>
    public KeyframeData ExtractPoseFromClip(AnimationClip clip, float time)
    {
        if (smplAnimator == null || clip == null) return new KeyframeData();
        
        // Store current state
        var currentState = smplAnimator.GetCurrentAnimatorStateInfo(0);
        
        // Sample the clip
        clip.SampleAnimation(smplAnimator.gameObject, time);
        
        // Extract the pose
        var frame = ExtractCurrentPose();
        
        return frame;
    }
    
    /// <summary>
    /// Extract pose from another animator's current state
    /// </summary>
    public KeyframeData ExtractPoseFromAnimator(Animator targetAnimator)
    {
        if (targetAnimator == null) return new KeyframeData();
        
        var frame = new KeyframeData();
        
        // Get root position
        Transform hips = targetAnimator.GetBoneTransform(HumanBodyBones.Hips);
        if (hips != null)
        {
            Vector3 pos = hips.position;
            frame.root_pos = new float[] { -pos.x, -pos.z, pos.y };
        }
        
        // Get joint rotations
        frame.joints = new List<JointRotation>();
        
        foreach (var kvp in boneMap)
        {
            Transform bone = targetAnimator.GetBoneTransform(kvp.Value);
            if (bone != null)
            {
                Quaternion rot = bone.localRotation;
                var jointRot = new JointRotation
                {
                    name = kvp.Key,
                    rot = new float[] { -rot.x, -rot.y, rot.z, rot.w }
                };
                frame.joints.Add(jointRot);
            }
        }
        
        return frame;
    }
    
    /// <summary>
    /// Send a ping to test connection
    /// </summary>
    public void Ping()
    {
        if (!isConnected) return;
        
        try
        {
            string json = "{\"type\":\"ping\"}\n";
            byte[] data = Encoding.UTF8.GetBytes(json);
            stream.Write(data, 0, data.Length);
        }
        catch (Exception e)
        {
            Debug.LogError($"[InbetweenRequester] Ping failed: {e.Message}");
        }
    }
    
    public void Disconnect()
    {
        isRunning = false;
        
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(100);
        }
        
        if (client != null)
        {
            client.Close();
            client = null;
        }
        
        stream = null;
        isConnected = false;
        currentState = InbetweenState.Idle;
        Debug.Log("[InbetweenRequester] Disconnected");
    }
    
    void OnDestroy()
    {
        Disconnect();
    }
    
    void OnGUI()
    {
        // Status display
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 12;
        
        string status = $"In-betweening: {currentState}";
        if (currentState == InbetweenState.Streaming)
        {
            status += $" ({receivedFrames}/{totalFrames})";
        }
        
        GUI.Label(new Rect(10, 10, 300, 20), status, style);
        
        string connStatus = isConnected ? $"Connected to {host}:{keyframePort}" : "Disconnected";
        GUI.Label(new Rect(10, 30, 300, 20), connStatus, style);
    }
    
    // --- Serializable classes for JSON ---
    
    [Serializable]
    public class KeyframeData
    {
        public float[] root_pos = new float[3];
        public List<JointRotation> joints = new List<JointRotation>();
    }
    
    [Serializable]
    public class JointRotation
    {
        public string name;
        public float[] rot; // [x, y, z, w]
    }
    
    [Serializable]
    public class InbetweenRequest
    {
        public string type;
        public KeyframeData start_frame;
        public KeyframeData end_frame;
        public string prompt;
        public int duration_frames;
        public string mode;
    }
    
    [Serializable]
    public class InbetweenResponse
    {
        public string type;
        public string status;
        public string message;
        public int duration_frames;
        public int total_frames;
        public int generation_time_ms;
        public bool success;
    }
}
