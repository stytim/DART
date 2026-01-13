using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;

/// <summary>
/// Records animation data received from DART streamer to a JSON file.
/// Attach this to the same GameObject as MotionReceiver to record incoming data.
/// </summary>
public class MotionRecorder : MonoBehaviour
{
    [Header("Recording Settings")]
    [Tooltip("File name for the recording (without extension)")]
    public string recordingFileName = "motion_recording";
    
    [Tooltip("Directory to save recordings (relative to Application.persistentDataPath)")]
    public string recordingDirectory = "Recordings";
    
    [Header("Controls")]
    public bool isRecording = false;
    
    [Header("Status (Read Only)")]
    [SerializeField] private int frameCount = 0;
    [SerializeField] private float recordingDuration = 0f;
    
    private List<RecordedFrame> recordedFrames = new List<RecordedFrame>();
    private float recordingStartTime = 0f;
    
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
    
    void Start()
    {
        // Ensure recording directory exists
        string fullPath = Path.Combine(Application.persistentDataPath, recordingDirectory);
        if (!Directory.Exists(fullPath))
        {
            Directory.CreateDirectory(fullPath);
        }
    }
    
    void Update()
    {
        if (isRecording)
        {
            recordingDuration = Time.time - recordingStartTime;
        }
    }
    
    /// <summary>
    /// Start recording animation frames
    /// </summary>
    public void StartRecording()
    {
        if (isRecording) return;
        
        recordedFrames.Clear();
        frameCount = 0;
        recordingDuration = 0f;
        recordingStartTime = Time.time;
        isRecording = true;
        
        Debug.Log("[MotionRecorder] Recording started.");
    }
    
    /// <summary>
    /// Stop recording and save to file
    /// </summary>
    public void StopRecording()
    {
        if (!isRecording) return;
        
        isRecording = false;
        SaveRecording();
        
        Debug.Log($"[MotionRecorder] Recording stopped. {frameCount} frames recorded.");
    }
    
    /// <summary>
    /// Record a single frame of animation data (call this from MotionReceiver)
    /// </summary>
    public void RecordFrame(string rawJson)
    {
        if (!isRecording) return;
        
        RecordedFrame frame = new RecordedFrame();
        frame.timestamp = Time.time - recordingStartTime;
        frame.rawJson = rawJson;
        
        recordedFrames.Add(frame);
        frameCount++;
    }
    
    private void SaveRecording()
    {
        RecordingData data = new RecordingData();
        data.createdAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        data.totalDuration = recordingDuration;
        data.totalFrames = frameCount;
        data.averageFps = frameCount > 0 ? frameCount / recordingDuration : 0f;
        data.frames = recordedFrames;
        
        string json = JsonUtility.ToJson(data, true);
        
        string fileName = $"{recordingFileName}_{DateTime.Now:yyyyMMdd_HHmmss}.json";
        string fullPath = Path.Combine(Application.persistentDataPath, recordingDirectory, fileName);
        
        File.WriteAllText(fullPath, json);
        
        Debug.Log($"[MotionRecorder] Saved recording to: {fullPath}");
    }
    
    /// <summary>
    /// Toggle recording on/off
    /// </summary>
    public void ToggleRecording()
    {
        if (isRecording)
            StopRecording();
        else
            StartRecording();
    }
}
