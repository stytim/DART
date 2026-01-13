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

public class MotionReceiver : MonoBehaviour
{
    [Header("Network Settings")]
    public string host = "127.0.0.1";
    public int port = 8080;
    
    [Header("Target Settings")]
    [Tooltip("Assign the Animator component of the character you want to animate.")]
    public Animator targetAnimator;
    public float positionScale = 1.0f; // DART is meters, Unity is meters. Adjust if scale is off.
    public bool debugMode = false;
    
    [Header("Recording (Optional)")]
    [Tooltip("Assign a MotionRecorder to record incoming data.")]
    public MotionRecorder recorder;
    
    private TcpClient client;
    private NetworkStream stream;
    private StreamReader reader;

    // Bone Mapping
    private Dictionary<string, HumanBodyBones> boneMap;

    void Start()
    {
        // Auto-assign if attached to the character
        if (targetAnimator == null)
        {
            targetAnimator = GetComponent<Animator>();
        }

        if (targetAnimator == null)
        {
            Debug.LogError("MotionReceiver: No Target Animator assigned! Please assign it in the Inspector.");
            return;
        }

        InitializeBoneMap();
        Connect();
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
                // Read all available lines (consume buffer to stay real-time)
                string lastLine = null;
                while (stream.DataAvailable) {
                    string line = reader.ReadLine();
                    if (!string.IsNullOrEmpty(line)) lastLine = line;
                }

                if (lastLine != null) {
                    if (debugMode) Debug.Log($"[DART] Received JSON: {lastLine}");
                    
                    // Record frame if recorder is attached
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
        if (targetAnimator == null) return;

        FrameData data = JsonUtility.FromJson<FrameData>(json);
        if (data == null) return;

        // Apply Root Position
        if (data.root_pos != null && data.root_pos.Length == 3)
        {
            Vector3 pos = new Vector3(data.root_pos[0], data.root_pos[1], data.root_pos[2]);
            if (debugMode) Debug.Log($"[DART] Applying Root Pos: {pos * positionScale}");
            // Apply to the GameObject holding the Animator
            targetAnimator.transform.position = pos * positionScale;
        }

        // Apply Joint Rotations
        int jointsUpdated = 0;
        foreach (var joint in data.joints)
        {
            if (boneMap.ContainsKey(joint.name))
            {
                Transform bone = targetAnimator.GetBoneTransform(boneMap[joint.name]);
                if (bone != null)
                {
                    Quaternion rot = new Quaternion(joint.rot[0], joint.rot[1], joint.rot[2], joint.rot[3]);
                    if (debugMode) Debug.Log($"[DART] Set {joint.name} to {rot.eulerAngles}");
                    bone.localRotation = rot;
                    jointsUpdated++;
                }
            }
        }
        if (debugMode) Debug.Log($"[DART] Updated {jointsUpdated} / {data.joints.Count} joints.");
    }

    void OnDestroy()
    {
        if (client != null) client.Close();
    }
}
