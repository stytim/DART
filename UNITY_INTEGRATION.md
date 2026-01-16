# DART to Unity Real-Time Streaming

This guide explains how to stream generated motion from DART to a Unity Humanoid Avatar in real-time.

## Architecture

```mermaid
graph LR
    DART[Python: DART Motion Gen] -->|TCP JSON Stream| Unity[Unity Client]
    subgraph DART Loop
        Gen[Motion Generation] --> Cache[Frame Cache]
        Cache --> Retarget[Retargeting Layer]
        Retarget --> Stream[Streamer]
    end
    subgraph Unity Client
        Recv[MotionReceiver.cs] --> Parse[JSON Parser]
        Parse --> Apply[Apply Rotations]
        Apply --> Avatar[Humanoid Avatar]
    end
```

## Data Flow
1.  **Generation**: DART generates motion (SMPL format).
2.  **Caching**: Root position, Root rotation, and Joint rotations are cached per frame.
3.  **Retargeting**:
    *   Coordinate System: Converted from Right-Handed (PyTorch3D) to Left-Handed (Unity).
    *   Format: Rotation Matrices converted to Quaternions.
    *   Mapping: SMPL joints mapped to Unity Humanoid Bone names.
4.  **Streaming**: JSON packets sent over TCP.
5.  **Reception**: Unity script parses JSON and applies `localRotation` to the avatar's Animator bones.

## Setup Instructions

### 1. Python Side
Run the demo with streaming enabled:
```bash
python -m mld.rollout_demo --denoiser_checkpoint <path_to_ckpt> --enable_streaming 1
```
Or use the provided script modification in `demos/run_demo.sh` (you may need to add `--enable_streaming 1` to the command line args there).

The server starts on `0.0.0.0:8080` by default (broadcasting to all interfaces).

### 2. Unity Side

1.  Create a fresh Unity 3D project.
2.  Import a humanoid character.
3.  Copy the `MotionReceiver.cs` file from the root of this repository into your Unity project's `Assets` folder.
4.  Create an empty GameObject in your scene (or select any object).
5.  Attach the `MotionReceiver` script to it.
6.  **Important**: In the Inspector, drag your Character (the GameObject with the Animator) into the `Target Animator` field of the `MotionReceiver` component.
7.  (Optional) Check **Debug Mode** to see console logs of received data.
8.  Press Play in Unity.

### Unity C# Script

*The script is now provided as `MotionReceiver.cs` in the root of the repo.*

## Recording and Playback

Two additional scripts are provided for recording and playing back animation data:

### Recording
1.  Copy `MotionRecorder.cs` to your Unity project.
2.  Add a `MotionRecorder` component to any GameObject.
3.  In the `MotionReceiver` Inspector, assign the `MotionRecorder` to the **Recorder** field.
4.  Use the `isRecording` checkbox or call `StartRecording()` / `StopRecording()` to control recording.
5.  Recordings are saved to `Application.persistentDataPath/Recordings/` as JSON files.

### Playback
1.  Copy `MotionPlayer.cs` to your Unity project.
2.  Add a `MotionPlayer` component to a GameObject.
3.  Assign your character's Animator to **Target Animator**.
4.  Set the **Recording File Path** to your saved recording.
5.  Call `LoadRecording()` then `Play()` to play back the animation.

---

## Motion In-Betweening (NEW)

Generate smooth transitions between two animation keyframes using text prompts. This is useful for seamlessly connecting pre-recorded animation sequences.

### Architecture

```mermaid
sequenceDiagram
    participant Unity
    participant Requester as MotionInbetweenRequester
    participant Server as DART Inbetween Server
    participant Receiver as MotionReceiverInbetween
    
    Unity->>Requester: RequestInbetween(startPose, endPose, "walk")
    Requester->>Server: JSON with keyframes + prompt
    Note over Server: Generate transition (~0.5-1s)
    loop For each frame
        Server->>Receiver: Stream animation frame
        Receiver->>Receiver: Incremental blend
    end
    Receiver->>Unity: OnInbetweenComplete event
```

### Python Server Setup

Start the in-betweening server:
```bash
# Fast mode (~0.5-1s generation time)
source demos/run_demo_inbetween_streaming.sh
```

The server listens on:
- **Port 8080**: Motion streaming to Unity
- **Port 8082**: Keyframe requests from Unity

### Unity Setup

1. Copy these scripts to your Unity project:
   - `MotionInbetweenRequester.cs` - Sends keyframe requests
   - `MotionReceiverInbetween.cs` - Receives and blends frames

2. Create GameObjects and attach scripts:
   ```
   InbetweenManager (GameObject)
   ├── MotionInbetweenRequester component
   └── MotionReceiverInbetween component
   ```

3. Configure the components in Inspector:
   - Set `Host` and `Port` to match your DART server
   - Assign your SMPL Animator to both components
   - Set `Blend Mode` to `Incremental` for smooth transitions

### Using with FBX Mocap Animations

Use the provided `AnimationTransitionController.cs` script to transition between FBX animation files:

1. Copy `AnimationTransitionController.cs` to your Unity project
2. Add it to the same GameObject as your other DART components
3. Drag your FBX animation clips into the **Animations** list
4. Set a **Default Prompt** like "walk forward" or "turn around"
5. Call transitions from your code:

```csharp
// Get reference to controller
AnimationTransitionController controller = GetComponent<AnimationTransitionController>();

// Transition by index
controller.TransitionToAnimation(0, "walk forward");

// Transition by clip name
controller.TransitionToAnimation("IdleToWalk", "start walking");

// Transition directly with clip
controller.TransitionToAnimation(myAnimationClip, "jump");
```

### Blend Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `Immediate` | Apply frames as they arrive | Low latency, may be jerky |
| `Incremental` | Smooth blend between frames | **Recommended** for smooth transitions |
| `BufferAndPlay` | Buffer all, then play | Highest quality, but delayed start |

### Events

```csharp
// MotionInbetweenRequester events
OnGenerationStarted    // Request sent, generation starting
OnFrameReceived(int frameIndex, int totalFrames)  // Progress updates
OnGenerationComplete(bool success)  // Generation finished

// MotionReceiverInbetween events
OnInbetweenStarted     // First frame received
OnInbetweenProgress(float progress)  // 0-1 progress
OnInbetweenComplete    // All frames played
```

---

## Troubleshooting
*   **Twisted Limbs**: Check the Coordinate System conversion in `utils/unity_streamer.py`. If limbs are twisted, try swapping the quaternion components (e.g., negate x or w).
*   **Latency**: The Unity script currently drains the buffer to play the latest frame. If it's too jittery, implement a jitter buffer (interpolate between frames).
