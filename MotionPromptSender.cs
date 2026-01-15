using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Net.Sockets;
using System.Text;
using System;

/// <summary>
/// Sends text prompts to the DART Python text-to-motion system.
/// Connect to the UnityPromptReceiver running on the server.
/// </summary>
public class MotionPromptSender : MonoBehaviour
{
    [Header("Network Settings")]
    [Tooltip("IP address of the DART server")]
    public string host = "127.0.0.1";
    
    [Tooltip("Port for prompt receiver (default 8081, different from motion streaming)")]
    public int port = 8081;
    
    [Tooltip("Auto-connect on start")]
    public bool autoConnect = true;
    
    [Header("UI References (Optional)")]
    [Tooltip("Input field for typing prompts")]
    public InputField promptInputField;
    
    [Tooltip("Button to send prompt")]
    public Button sendButton;
    
    [Header("Keyboard Controls")]
    [Tooltip("Key to open prompt input (when no UI)")]
    public KeyCode promptKey = KeyCode.Return;
    
    [Header("Status")]
    [SerializeField] private bool isConnected = false;
    [SerializeField] private string lastPrompt = "";
    
    private TcpClient client;
    private NetworkStream stream;
    private bool isTyping = false;
    private string currentInput = "";
    
    void Start()
    {
        if (sendButton != null)
        {
            sendButton.onClick.AddListener(OnSendButtonClick);
        }
        
        if (autoConnect)
        {
            Connect();
        }
    }
    
    void Update()
    {
        // Simple keyboard input when no UI is assigned
        if (promptInputField == null)
        {
            HandleKeyboardInput();
        }
        else
        {
            // Send on Enter key when input field is focused
            if (Input.GetKeyDown(KeyCode.Return) && promptInputField.isFocused)
            {
                SendPromptFromInputField();
            }
        }
    }
    
    void HandleKeyboardInput()
    {
        if (!isTyping)
        {
            // Press Enter to start typing
            if (Input.GetKeyDown(promptKey))
            {
                isTyping = true;
                currentInput = "";
                Debug.Log("[MotionPromptSender] Type your prompt and press Enter...");
            }
        }
        else
        {
            // Capture text input
            foreach (char c in Input.inputString)
            {
                if (c == '\b') // Backspace
                {
                    if (currentInput.Length > 0)
                    {
                        currentInput = currentInput.Substring(0, currentInput.Length - 1);
                    }
                }
                else if (c == '\n' || c == '\r') // Enter
                {
                    if (!string.IsNullOrEmpty(currentInput))
                    {
                        SendPrompt(currentInput);
                        currentInput = "";
                    }
                    isTyping = false;
                }
                else
                {
                    currentInput += c;
                }
            }
        }
    }
    
    void OnGUI()
    {
        // Show typing UI when in keyboard mode
        if (isTyping && promptInputField == null)
        {
            GUIStyle style = new GUIStyle(GUI.skin.box);
            style.fontSize = 16;
            style.alignment = TextAnchor.MiddleLeft;
            
            string display = $"Prompt: {currentInput}_";
            GUI.Box(new Rect(10, Screen.height - 50, Screen.width - 20, 40), display, style);
        }
        
        // Connection status
        GUIStyle statusStyle = new GUIStyle(GUI.skin.label);
        statusStyle.fontSize = 12;
        string status = isConnected ? $"Connected to {host}:{port}" : "Disconnected (press C to connect)";
        GUI.Label(new Rect(10, Screen.height - 80, 300, 20), status, statusStyle);
        
        // C to connect
        if (Input.GetKeyDown(KeyCode.C) && !isConnected)
        {
            Connect();
        }
    }
    
    /// <summary>
    /// Connect to the DART prompt receiver
    /// </summary>
    public void Connect()
    {
        if (isConnected) return;
        
        try
        {
            client = new TcpClient();
            client.BeginConnect(host, port, OnConnected, null);
            Debug.Log($"[MotionPromptSender] Connecting to {host}:{port}...");
        }
        catch (Exception e)
        {
            Debug.LogError($"[MotionPromptSender] Connection failed: {e.Message}");
        }
    }
    
    void OnConnected(IAsyncResult ar)
    {
        try
        {
            client.EndConnect(ar);
            stream = client.GetStream();
            isConnected = true;
            Debug.Log($"[MotionPromptSender] Connected to DART prompt receiver!");
        }
        catch (Exception e)
        {
            Debug.LogError($"[MotionPromptSender] Connection error: {e.Message}");
            isConnected = false;
        }
    }
    
    /// <summary>
    /// Send a text prompt to the DART system
    /// </summary>
    public void SendPrompt(string prompt)
    {
        if (string.IsNullOrEmpty(prompt)) return;
        
        if (!isConnected || stream == null)
        {
            Debug.LogWarning("[MotionPromptSender] Not connected. Call Connect() first.");
            return;
        }
        
        try
        {
            // Create JSON message
            var message = new PromptMessage { prompt = prompt };
            string json = JsonUtility.ToJson(message) + "\n";
            
            byte[] data = Encoding.UTF8.GetBytes(json);
            stream.Write(data, 0, data.Length);
            
            lastPrompt = prompt;
            Debug.Log($"[MotionPromptSender] Sent prompt: '{prompt}'");
        }
        catch (Exception e)
        {
            Debug.LogError($"[MotionPromptSender] Send error: {e.Message}");
            isConnected = false;
        }
    }
    
    void OnSendButtonClick()
    {
        SendPromptFromInputField();
    }
    
    void SendPromptFromInputField()
    {
        if (promptInputField != null && !string.IsNullOrEmpty(promptInputField.text))
        {
            SendPrompt(promptInputField.text);
            promptInputField.text = "";
        }
    }
    
    /// <summary>
    /// Disconnect from the server
    /// </summary>
    public void Disconnect()
    {
        if (client != null)
        {
            client.Close();
            client = null;
        }
        stream = null;
        isConnected = false;
        Debug.Log("[MotionPromptSender] Disconnected");
    }
    
    void OnDestroy()
    {
        Disconnect();
    }
    
    [System.Serializable]
    private class PromptMessage
    {
        public string prompt;
    }
}
