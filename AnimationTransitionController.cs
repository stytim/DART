using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Controls avatar animation transitions using DART motion in-betweening.
/// Works with FBX mocap animation files to create smooth AI-generated transitions.
/// 
/// Setup:
/// 1. Add this script to a GameObject
/// 2. Assign the MotionInbetweenRequester and MotionReceiverInbetween components
/// 3. Assign your SMPL Avatar's Animator
/// 4. Add your FBX animation clips to the Animations list
/// 5. Call TransitionToAnimation() with the animation index and a text prompt
/// </summary>
public class AnimationTransitionController : MonoBehaviour
{
    [Header("DART Components")]
    [Tooltip("Reference to the MotionInbetweenRequester component")]
    public MotionInbetweenRequester inbetweenRequester;
    
    [Tooltip("Reference to the MotionReceiverInbetween component")]
    public MotionReceiverInbetween motionReceiver;
    
    [Header("Avatar")]
    [Tooltip("The Animator component of your avatar (SMPL or retargeted)")]
    public Animator avatarAnimator;
    
    [Header("Mocap Animations")]
    [Tooltip("List of FBX animation clips to transition between")]
    public List<AnimationClip> animations = new List<AnimationClip>();
    
    [Tooltip("Sample time in the target animation to extract the end pose (seconds)")]
    public float targetPoseSampleTime = 0.0f;
    
    [Header("Transition Settings")]
    [Tooltip("Duration of generated transition in frames (30fps). 60 = 2 seconds")]
    public int transitionDurationFrames = 60;
    
    [Tooltip("Default text prompt for transitions")]
    public string defaultPrompt = "walk forward";
    
    [Header("Playback Settings")]
    [Tooltip("Automatically play target animation after transition completes")]
    public bool autoPlayAfterTransition = true;
    
    [Tooltip("Blend time when starting target animation (seconds)")]
    public float animationBlendTime = 0.25f;
    
    [Header("Status")]
    [SerializeField] private TransitionState currentState = TransitionState.Idle;
    [SerializeField] private int currentAnimationIndex = -1;
    [SerializeField] private int pendingAnimationIndex = -1;
    [SerializeField] private float transitionProgress = 0f;
    
    // Events
    public event System.Action<int> OnTransitionStarted;      // animationIndex
    public event System.Action<float> OnTransitionProgress;   // progress 0-1
    public event System.Action<int> OnTransitionComplete;     // animationIndex
    public event System.Action<string> OnTransitionFailed;    // error message
    
    public enum TransitionState
    {
        Idle,
        RequestingTransition,
        GeneratingTransition,
        PlayingTransition,
        Complete
    }
    
    public TransitionState State => currentState;
    public float Progress => transitionProgress;
    public int CurrentAnimationIndex => currentAnimationIndex;
    
    void Start()
    {
        ValidateSetup();
        SubscribeToEvents();
    }
    
    void ValidateSetup()
    {
        if (inbetweenRequester == null)
        {
            Debug.LogError("[AnimationTransitionController] MotionInbetweenRequester not assigned!");
        }
        if (motionReceiver == null)
        {
            Debug.LogError("[AnimationTransitionController] MotionReceiverInbetween not assigned!");
        }
        if (avatarAnimator == null)
        {
            Debug.LogError("[AnimationTransitionController] Avatar Animator not assigned!");
        }
    }
    
    void SubscribeToEvents()
    {
        if (inbetweenRequester != null)
        {
            inbetweenRequester.OnGenerationStarted += OnDARTGenerationStarted;
            inbetweenRequester.OnGenerationComplete += OnDARTGenerationComplete;
        }
        
        if (motionReceiver != null)
        {
            motionReceiver.OnInbetweenStarted += OnInbetweenPlaybackStarted;
            motionReceiver.OnInbetweenProgress += OnInbetweenPlaybackProgress;
            motionReceiver.OnInbetweenComplete += OnInbetweenPlaybackComplete;
        }
    }
    
    void OnDestroy()
    {
        if (inbetweenRequester != null)
        {
            inbetweenRequester.OnGenerationStarted -= OnDARTGenerationStarted;
            inbetweenRequester.OnGenerationComplete -= OnDARTGenerationComplete;
        }
        
        if (motionReceiver != null)
        {
            motionReceiver.OnInbetweenStarted -= OnInbetweenPlaybackStarted;
            motionReceiver.OnInbetweenProgress -= OnInbetweenPlaybackProgress;
            motionReceiver.OnInbetweenComplete -= OnInbetweenPlaybackComplete;
        }
    }
    
    /// <summary>
    /// Transition from current pose to a specific animation by index.
    /// </summary>
    /// <param name="animationIndex">Index in the animations list</param>
    /// <param name="prompt">Text prompt describing the transition (e.g., "walk forward", "turn around")</param>
    public void TransitionToAnimation(int animationIndex, string prompt = null)
    {
        if (currentState != TransitionState.Idle && currentState != TransitionState.Complete)
        {
            Debug.LogWarning("[AnimationTransitionController] Transition already in progress!");
            return;
        }
        
        if (animationIndex < 0 || animationIndex >= animations.Count)
        {
            Debug.LogError($"[AnimationTransitionController] Invalid animation index: {animationIndex}");
            OnTransitionFailed?.Invoke($"Invalid animation index: {animationIndex}");
            return;
        }
        
        AnimationClip targetClip = animations[animationIndex];
        if (targetClip == null)
        {
            Debug.LogError($"[AnimationTransitionController] Animation at index {animationIndex} is null!");
            OnTransitionFailed?.Invoke($"Animation at index {animationIndex} is null");
            return;
        }
        
        pendingAnimationIndex = animationIndex;
        currentState = TransitionState.RequestingTransition;
        transitionProgress = 0f;
        
        string usePrompt = string.IsNullOrEmpty(prompt) ? defaultPrompt : prompt;
        
        Debug.Log($"[AnimationTransitionController] Requesting transition to '{targetClip.name}' with prompt '{usePrompt}'");
        
        // Request in-betweening from current pose to target animation pose
        inbetweenRequester.RequestInbetween(
            targetClip,
            targetPoseSampleTime,
            usePrompt,
            transitionDurationFrames
        );
        
        OnTransitionStarted?.Invoke(animationIndex);
    }
    
    /// <summary>
    /// Transition to animation by name (searches the animations list).
    /// </summary>
    public void TransitionToAnimation(string animationName, string prompt = null)
    {
        int index = animations.FindIndex(clip => clip != null && clip.name == animationName);
        if (index >= 0)
        {
            TransitionToAnimation(index, prompt);
        }
        else
        {
            Debug.LogError($"[AnimationTransitionController] Animation '{animationName}' not found!");
            OnTransitionFailed?.Invoke($"Animation '{animationName}' not found");
        }
    }
    
    /// <summary>
    /// Transition to animation using the clip directly.
    /// </summary>
    public void TransitionToAnimation(AnimationClip clip, string prompt = null, float sampleTime = -1f)
    {
        if (clip == null)
        {
            Debug.LogError("[AnimationTransitionController] Clip is null!");
            return;
        }
        
        if (currentState != TransitionState.Idle && currentState != TransitionState.Complete)
        {
            Debug.LogWarning("[AnimationTransitionController] Transition already in progress!");
            return;
        }
        
        // Add to list if not already present
        int index = animations.IndexOf(clip);
        if (index < 0)
        {
            animations.Add(clip);
            index = animations.Count - 1;
        }
        
        float useSampleTime = sampleTime >= 0 ? sampleTime : targetPoseSampleTime;
        pendingAnimationIndex = index;
        currentState = TransitionState.RequestingTransition;
        transitionProgress = 0f;
        
        string usePrompt = string.IsNullOrEmpty(prompt) ? defaultPrompt : prompt;
        
        inbetweenRequester.RequestInbetween(clip, useSampleTime, usePrompt, transitionDurationFrames);
        OnTransitionStarted?.Invoke(index);
    }
    
    /// <summary>
    /// Cancel the current transition and return to idle state.
    /// </summary>
    public void CancelTransition()
    {
        if (currentState == TransitionState.Idle) return;
        
        Debug.Log("[AnimationTransitionController] Cancelling transition");
        motionReceiver?.ReturnToStreaming();
        currentState = TransitionState.Idle;
        pendingAnimationIndex = -1;
    }
    
    /// <summary>
    /// Play an animation directly without transition (immediate switch).
    /// </summary>
    public void PlayAnimationImmediate(int animationIndex)
    {
        if (animationIndex < 0 || animationIndex >= animations.Count) return;
        
        AnimationClip clip = animations[animationIndex];
        if (clip == null) return;
        
        PlayClipOnAvatar(clip);
        currentAnimationIndex = animationIndex;
    }
    
    // --- Event Handlers ---
    
    void OnDARTGenerationStarted()
    {
        currentState = TransitionState.GeneratingTransition;
        Debug.Log("[AnimationTransitionController] DART generation started");
    }
    
    void OnDARTGenerationComplete(bool success)
    {
        if (!success)
        {
            Debug.LogError("[AnimationTransitionController] DART generation failed!");
            currentState = TransitionState.Idle;
            pendingAnimationIndex = -1;
            OnTransitionFailed?.Invoke("DART generation failed");
        }
        // If success, wait for playback to complete
    }
    
    void OnInbetweenPlaybackStarted()
    {
        currentState = TransitionState.PlayingTransition;
        Debug.Log("[AnimationTransitionController] Transition playback started");
    }
    
    void OnInbetweenPlaybackProgress(float progress)
    {
        transitionProgress = progress;
        OnTransitionProgress?.Invoke(progress);
    }
    
    void OnInbetweenPlaybackComplete()
    {
        Debug.Log("[AnimationTransitionController] Transition playback complete");
        currentState = TransitionState.Complete;
        currentAnimationIndex = pendingAnimationIndex;
        
        OnTransitionComplete?.Invoke(currentAnimationIndex);
        
        // Auto-play target animation if enabled
        if (autoPlayAfterTransition && currentAnimationIndex >= 0)
        {
            AnimationClip clip = animations[currentAnimationIndex];
            if (clip != null)
            {
                StartCoroutine(PlayAnimationAfterDelay(clip, 0.1f));
            }
        }
        
        // Reset to idle after a short delay
        StartCoroutine(ResetToIdleAfterDelay(0.5f));
    }
    
    IEnumerator PlayAnimationAfterDelay(AnimationClip clip, float delay)
    {
        yield return new WaitForSeconds(delay);
        PlayClipOnAvatar(clip);
    }
    
    IEnumerator ResetToIdleAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        currentState = TransitionState.Idle;
    }
    
    void PlayClipOnAvatar(AnimationClip clip)
    {
        if (avatarAnimator == null || clip == null) return;
        
        // Check if clip is Legacy (required for Animation component)
        if (!clip.legacy)
        {
            Debug.LogWarning($"[AnimationTransitionController] AnimationClip '{clip.name}' is not marked as Legacy. " +
                "To fix: Select the FBX file in Project window → Inspector → Rig tab → Animation Type = Legacy → Apply. " +
                "Alternatively, add states to your Animator Controller matching the clip names.");
        }
        
        // Try legacy Animation component first (most flexible for playing arbitrary clips)
        Animation legacyAnim = avatarAnimator.GetComponent<Animation>();
        if (legacyAnim != null && clip.legacy)
        {
            // Add the clip to the Animation component if not already present
            if (legacyAnim.GetClip(clip.name) == null)
            {
                legacyAnim.AddClip(clip, clip.name);
                Debug.Log($"[AnimationTransitionController] Added clip '{clip.name}' to Animation component");
            }
            
            // Play the clip
            legacyAnim.CrossFade(clip.name, animationBlendTime);
            Debug.Log($"[AnimationTransitionController] Playing animation via Animation component: {clip.name}");
            return;
        }
        
        // Fallback: try Animator Controller (requires matching state name)
        if (avatarAnimator.runtimeAnimatorController != null)
        {
            // Check if state exists before trying to play
            bool stateExists = false;
            foreach (var animClip in avatarAnimator.runtimeAnimatorController.animationClips)
            {
                if (animClip.name == clip.name)
                {
                    stateExists = true;
                    break;
                }
            }
            
            if (stateExists)
            {
                avatarAnimator.Play(clip.name, 0, 0f);
                Debug.Log($"[AnimationTransitionController] Playing animation via Animator: {clip.name}");
                return;
            }
        }
        
        // Neither method worked - skip auto-play
        Debug.LogWarning($"[AnimationTransitionController] Could not auto-play '{clip.name}'. " +
            "Either mark the FBX as Legacy, or add a state with this name to your Animator Controller.");
    }
    
    // --- Inspector Helpers ---
    
    [ContextMenu("Test Transition to First Animation")]
    void TestTransitionToFirst()
    {
        if (animations.Count > 0)
        {
            TransitionToAnimation(0, defaultPrompt);
        }
    }
    
    [ContextMenu("Cancel Current Transition")]
    void TestCancelTransition()
    {
        CancelTransition();
    }
    
    void OnGUI()
    {
        // Simple debug UI
        GUIStyle style = new GUIStyle(GUI.skin.box);
        style.fontSize = 14;
        
        string status = $"Transition: {currentState}";
        if (currentState == TransitionState.PlayingTransition)
        {
            status += $" ({transitionProgress * 100:F0}%)";
        }
        
        GUI.Box(new Rect(10, Screen.height - 60, 300, 50), status, style);
    }
}
