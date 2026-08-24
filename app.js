/**
 * StudyEye AI - Automated Real-Time Study Focus Monitor
 * ------------------------------------------------------
 * Client-side Computer Vision pipeline using MediaPipe Face Mesh,
 * Eye-Aspect-Ratio (EAR) temporal tracking, and Web Speech API alerts.
 */

// ==============================================================================
// CONFIGURATION & CONSTANTS
// ==============================================================================
const CONFIG = {
  // Landmarks for Left Eye: [p1, p2, p3, p4, p5, p6]
  // p1 (outer), p4 (inner), (p2, p6) and (p3, p5) are vertical pairs
  LEFT_EYE_LANDMARKS: [33, 160, 158, 133, 153, 144],

  // Landmarks for Right Eye: [p1, p2, p3, p4, p5, p6]
  // p1 (inner), p4 (outer), (p2, p6) and (p3, p5) are vertical pairs
  RIGHT_EYE_LANDMARKS: [362, 385, 387, 263, 373, 380],

  // Iris Landmark indices (MediaPipe 468+ refinement)
  LEFT_IRIS: [468, 469, 470, 471],
  RIGHT_IRIS: [473, 474, 475, 476],

  ALERT_MESSAGE: "Wake up and focus on studying!",
  SPEECH_COOLDOWN_MS: 3500
};

// ==============================================================================
// STATE MANAGEMENT
// ==============================================================================
const state = {
  isRunning: false,
  isMirrored: true,
  showWireframe: true,
  
  // User Configurable Parameters
  earThreshold: 0.20,
  sleepLimitSeconds: 1.5,
  voiceAlertEnabled: true,
  audioBeepEnabled: true,

  // Temporal Sleep Tracker
  closedStartTime: null,
  elapsedSleepTime: 0.0,
  isAlertActive: false,
  lastSpeechTime: 0,

  // Session Analytics
  sessionStartTime: null,
  totalSessionSeconds: 0,
  alertCount: 0,
  drowsyFrames: 0,
  totalFrames: 0,

  // Performance Telemetry
  fps: 0,
  lastFrameTime: performance.now()
};

// ==============================================================================
// DOM ELEMENTS
// ==============================================================================
const elements = {
  video: document.getElementById('webcam'),
  canvas: document.getElementById('outputCanvas'),
  ctx: document.getElementById('outputCanvas').getContext('2d'),
  startPlaceholder: document.getElementById('startPlaceholder'),
  startCameraBtn: document.getElementById('startCameraBtn'),
  toggleMonitorBtn: document.getElementById('toggleMonitorBtn'),
  toggleIcon: document.getElementById('toggleIcon'),
  toggleText: document.getElementById('toggleText'),
  flipCameraBtn: document.getElementById('flipCameraBtn'),
  toggleLandmarksBtn: document.getElementById('toggleLandmarksBtn'),
  landmarkToggleText: document.getElementById('landmarkToggleText'),
  testVoiceBtn: document.getElementById('testVoiceBtn'),
  resetStatsBtn: document.getElementById('resetStatsBtn'),

  // Overlays
  faceMissingWarning: document.getElementById('faceMissingWarning'),
  sleepAlertOverlay: document.getElementById('sleepAlertOverlay'),

  // HUD
  fpsDisplay: document.getElementById('fpsDisplay'),
  avgEarDisplay: document.getElementById('avgEarDisplay'),
  lrEarDisplay: document.getElementById('lrEarDisplay'),
  sleepTimerText: document.getElementById('sleepTimerText'),
  sleepProgressFill: document.getElementById('sleepProgressFill'),
  systemStatusBadge: document.getElementById('systemStatusBadge'),
  systemStatusText: document.getElementById('systemStatusText'),

  // Gauge & Analytics
  earMeterFill: document.getElementById('earMeterFill'),
  earThresholdMarker: document.getElementById('earThresholdMarker'),
  thresholdLabel: document.getElementById('thresholdLabel'),
  earStateBadge: document.getElementById('earStateBadge'),
  sessionTimeDisplay: document.getElementById('sessionTimeDisplay'),
  focusScoreDisplay: document.getElementById('focusScoreDisplay'),
  alertCountDisplay: document.getElementById('alertCountDisplay'),
  stateBadgeDisplay: document.getElementById('stateBadgeDisplay'),

  // Sliders & Checkboxes
  earThresholdSlider: document.getElementById('earThresholdSlider'),
  earThresholdVal: document.getElementById('earThresholdVal'),
  sleepLimitSlider: document.getElementById('sleepLimitSlider'),
  sleepLimitVal: document.getElementById('sleepLimitVal'),
  voiceAlertToggle: document.getElementById('voiceAlertToggle'),
  audioBeepToggle: document.getElementById('audioBeepToggle'),
  soundSelect: document.getElementById('soundSelect'),
  previewSoundBtn: document.getElementById('previewSoundBtn')
};

// Custom Audio Library
const soundLibrary = {
  'alert.webm': new Audio('alert.webm'),
  'alert2.webm': new Audio('alert2.webm')
};

// Preload audio files
Object.values(soundLibrary).forEach(audio => {
  audio.preload = 'auto';
});

function playAlertSound() {
  if (!state.audioBeepEnabled) return;
  
  const selectedSound = elements.soundSelect ? elements.soundSelect.value : 'alert2.webm';

  if (selectedSound === 'chime') {
    playAlertChime();
    return;
  }

  let audioFile = selectedSound;
  if (selectedSound === 'random') {
    const keys = ['alert.webm', 'alert2.webm'];
    audioFile = keys[Math.floor(Math.random() * keys.length)];
  }

  const audioToPlay = soundLibrary[audioFile] || soundLibrary['alert2.webm'];
  if (audioToPlay) {
    try {
      audioToPlay.currentTime = 0;
      const playPromise = audioToPlay.play();
      if (playPromise !== undefined) {
        playPromise.catch(error => {
          console.warn("Custom audio playback failed, falling back to synthesizer chime:", error);
          playAlertChime();
        });
      }
    } catch (e) {
      console.warn("Audio play error:", e);
      playAlertChime();
    }
  } else {
    playAlertChime();
  }
}

// Audio Synthesizer Fallback Context for Chime Alert
let audioCtx = null;

function playAlertChime() {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.35); // A4

    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.35);
  } catch (e) {
    console.warn("Audio chime failed:", e);
  }
}

// ==============================================================================
// SPEECH & SOUND ALERT MANAGER
// ==============================================================================
function triggerVoiceAlert(message) {
  const now = Date.now();
  if (now - state.lastSpeechTime < CONFIG.SPEECH_COOLDOWN_MS) {
    return;
  }

  // Play custom alert sound
  playAlertSound();

  // Trigger TTS in parallel if enabled
  if (state.voiceAlertEnabled && 'speechSynthesis' in window) {
    window.speechSynthesis.cancel(); // Cancel any ongoing speech
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Siri')));
    if (englishVoice) {
      utterance.voice = englishVoice;
    }

    window.speechSynthesis.speak(utterance);
  }

  state.lastSpeechTime = now;
}

// ==============================================================================
// MATHEMATICAL COMPUTATIONS (EAR)
// ==============================================================================
function euclideanDistance(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function calculateEAR(landmarks, indices) {
  const p1 = landmarks[indices[0]]; // Outer corner
  const p2 = landmarks[indices[1]]; // Top 1
  const p3 = landmarks[indices[2]]; // Top 2
  const p4 = landmarks[indices[3]]; // Inner corner
  const p5 = landmarks[indices[4]]; // Bottom 2
  const p6 = landmarks[indices[5]]; // Bottom 1

  const vertical1 = euclideanDistance(p2, p6);
  const vertical2 = euclideanDistance(p3, p5);
  const horizontal = euclideanDistance(p1, p4);

  if (horizontal < 1e-6) return 0.0;
  return (vertical1 + vertical2) / (2.0 * horizontal);
}

// ==============================================================================
// MEDIAPIPE FACE MESH & CAMERA INITIALIZATION
// ==============================================================================
let faceMesh = null;
let camera = null;

function initializeFaceMesh() {
  faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });

  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  faceMesh.onResults(onResults);
}

async function startCamera() {
  if (!faceMesh) {
    initializeFaceMesh();
  }

  updateSystemStatus('Initializing...', 'status-ready');

  try {
    camera = new Camera(elements.video, {
      onFrame: async () => {
        if (state.isRunning) {
          await faceMesh.send({ image: elements.video });
        }
      },
      width: 1280,
      height: 720
    });

    await camera.start();

    state.isRunning = true;
    state.sessionStartTime = Date.now();
    elements.startPlaceholder.classList.add('hidden');
    elements.toggleIcon.className = 'fa-solid fa-pause';
    elements.toggleText.textContent = 'Pause Monitor';
    elements.toggleMonitorBtn.classList.remove('btn-primary');
    elements.toggleMonitorBtn.classList.add('btn-secondary');

    updateSystemStatus('Active & Monitoring', 'status-active');
  } catch (err) {
    console.error("Camera access error:", err);
    alert("Camera permission denied or camera unavailable. Please allow camera permissions in your browser.");
    updateSystemStatus('Camera Error', 'status-danger');
  }
}

function stopCamera() {
  state.isRunning = false;
  elements.toggleIcon.className = 'fa-solid fa-play';
  elements.toggleText.textContent = 'Resume Monitor';
  elements.toggleMonitorBtn.classList.remove('btn-secondary');
  elements.toggleMonitorBtn.classList.add('btn-primary');
  updateSystemStatus('Paused', 'status-warning');
}

// ==============================================================================
// FRAME RENDERING & INFERENCE PIPELINE
// ==============================================================================
function onResults(results) {
  const canvas = elements.canvas;
  const ctx = elements.ctx;

  // Sync canvas size with video feed
  if (canvas.width !== elements.video.videoWidth && elements.video.videoWidth > 0) {
    canvas.width = elements.video.videoWidth;
    canvas.height = elements.video.videoHeight;
  }

  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw incoming video frame onto canvas
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  // Calculate FPS
  const now = performance.now();
  const dt = (now - state.lastFrameTime) / 1000;
  state.lastFrameTime = now;
  if (dt > 0) {
    state.fps = Math.round(0.9 * state.fps + 0.1 * (1.0 / dt));
    elements.fpsDisplay.textContent = state.fps.toFixed(0);
  }

  state.totalFrames++;

  // --------------------------------------------------------------------------
  // SCENARIO A: Face Missing / Covered / Out of Frame
  // --------------------------------------------------------------------------
  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    state.closedStartTime = null;
    state.elapsedSleepTime = 0.0;
    state.isAlertActive = false;

    elements.faceMissingWarning.classList.remove('hidden');
    elements.sleepAlertOverlay.classList.add('hidden');

    updateSleepProgress(0.0);
    updateSystemStatus('Face Not Detected', 'status-danger');
    updateHUDValues(0, 0, 0);

    ctx.restore();
    return;
  }

  // Hide face missing warning
  elements.faceMissingWarning.classList.add('hidden');

  // --------------------------------------------------------------------------
  // SCENARIO B: Face Detected - Compute EAR & Temporal Tracker
  // --------------------------------------------------------------------------
  const landmarks = results.multiFaceLandmarks[0];

  const leftEAR = calculateEAR(landmarks, CONFIG.LEFT_EYE_LANDMARKS);
  const rightEAR = calculateEAR(landmarks, CONFIG.RIGHT_EYE_LANDMARKS);
  const avgEAR = (leftEAR + rightEAR) / 2.0;

  updateHUDValues(avgEAR, leftEAR, rightEAR);

  // Evaluate Drowsiness State
  if (avgEAR < state.earThreshold) {
    state.drowsyFrames++;
    if (state.closedStartTime === null) {
      state.closedStartTime = performance.now();
    }
    state.elapsedSleepTime = (performance.now() - state.closedStartTime) / 1000;

    // Check if sleep duration exceeded limit
    if (state.elapsedSleepTime >= state.sleepLimitSeconds) {
      if (!state.isAlertActive) {
        state.alertCount++;
        elements.alertCountDisplay.textContent = state.alertCount;
      }
      state.isAlertActive = true;
      elements.sleepAlertOverlay.classList.remove('hidden');
      updateSystemStatus('ALERT: DROWSY / SLEEPING', 'status-danger');
      triggerVoiceAlert(CONFIG.ALERT_MESSAGE);
    } else {
      state.isAlertActive = false;
      elements.sleepAlertOverlay.classList.add('hidden');
      updateSystemStatus('WARNING: EYES CLOSING', 'status-warning');
    }
  } else {
    // Eyes are open and focused
    state.closedStartTime = null;
    state.elapsedSleepTime = 0.0;
    state.isAlertActive = false;
    elements.sleepAlertOverlay.classList.add('hidden');
    updateSystemStatus('Focused & Alert', 'status-active');
  }

  updateSleepProgress(state.elapsedSleepTime);

  // Render Visual Contours on Canvas
  if (state.showWireframe) {
    drawEyeContour(ctx, landmarks, CONFIG.LEFT_EYE_LANDMARKS, canvas.width, canvas.height);
    drawEyeContour(ctx, landmarks, CONFIG.RIGHT_EYE_LANDMARKS, canvas.width, canvas.height);
  }

  ctx.restore();
}

function drawEyeContour(ctx, landmarks, indices, width, height) {
  const points = indices.map(idx => ({
    x: landmarks[idx].x * width,
    y: landmarks[idx].y * height
  }));

  const color = state.isAlertActive ? '#ef4444' : (state.elapsedSleepTime > 0 ? '#f59e0b' : '#00f2fe');

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.closePath();

  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.stroke();

  // Draw points
  ctx.shadowBlur = 0;
  ctx.fillStyle = '#ffffff';
  points.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 2.5, 0, 2 * Math.PI);
    ctx.fill();
  });
}

// ==============================================================================
// UI & TELEMETRY UPDATES
// ==============================================================================
function updateHUDValues(avgEAR, leftEAR, rightEAR) {
  elements.avgEarDisplay.textContent = avgEAR.toFixed(2);
  elements.lrEarDisplay.textContent = `${leftEAR.toFixed(2)} / ${rightEAR.toFixed(2)}`;

  // Update Gauge Fill (scale 0.0 -> 0.5 to 0% -> 100%)
  const gaugePercent = Math.min(100, Math.max(0, (avgEAR / 0.50) * 100));
  elements.earMeterFill.style.width = `${gaugePercent}%`;

  if (avgEAR < state.earThreshold) {
    elements.earStateBadge.textContent = 'Closed';
    elements.earStateBadge.style.color = '#ef4444';
    elements.earStateBadge.style.background = 'rgba(239, 68, 68, 0.15)';
    elements.earStateBadge.style.borderColor = 'rgba(239, 68, 68, 0.3)';
    elements.earMeterFill.style.background = 'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)';
  } else {
    elements.earStateBadge.textContent = 'Normal';
    elements.earStateBadge.style.color = '#10b981';
    elements.earStateBadge.style.background = 'rgba(16, 185, 129, 0.15)';
    elements.earStateBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    elements.earMeterFill.style.background = 'linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)';
  }
}

function updateSleepProgress(elapsed) {
  const limit = state.sleepLimitSeconds;
  const ratio = Math.min(1.0, elapsed / limit);
  const percent = ratio * 100;

  elements.sleepTimerText.textContent = `${elapsed.toFixed(1)}s / ${limit.toFixed(1)}s`;
  elements.sleepProgressFill.style.width = `${percent}%`;

  elements.sleepProgressFill.classList.remove('warning', 'danger');
  if (ratio >= 1.0) {
    elements.sleepProgressFill.classList.add('danger');
  } else if (ratio > 0.4) {
    elements.sleepProgressFill.classList.add('warning');
  }
}

function updateSystemStatus(text, statusClass) {
  elements.systemStatusText.textContent = text;
  elements.systemStatusBadge.className = `status-pill ${statusClass}`;
  
  if (text.includes('ALERT')) {
    elements.stateBadgeDisplay.textContent = 'Drowsy';
    elements.stateBadgeDisplay.className = 'stat-value text-danger';
  } else if (text.includes('WARNING')) {
    elements.stateBadgeDisplay.textContent = 'Eyes Closing';
    elements.stateBadgeDisplay.className = 'stat-value text-warning';
  } else if (text.includes('Face Not Detected')) {
    elements.stateBadgeDisplay.textContent = 'Away / Hidden';
    elements.stateBadgeDisplay.className = 'stat-value text-danger';
  } else if (state.isRunning) {
    elements.stateBadgeDisplay.textContent = 'Focused';
    elements.stateBadgeDisplay.className = 'stat-value text-success';
  } else {
    elements.stateBadgeDisplay.textContent = 'Standby';
    elements.stateBadgeDisplay.className = 'stat-value';
  }
}

// Session Timer & Analytics Loop
setInterval(() => {
  if (!state.isRunning || !state.sessionStartTime) return;

  state.totalSessionSeconds = Math.floor((Date.now() - state.sessionStartTime) / 1000);
  const mins = Math.floor(state.totalSessionSeconds / 60).toString().padStart(2, '0');
  const secs = (state.totalSessionSeconds % 60).toString().padStart(2, '0');
  elements.sessionTimeDisplay.textContent = `${mins}:${secs}`;

  // Focus Score = (1 - drowsyFrames / totalFrames) * 100
  if (state.totalFrames > 0) {
    const score = Math.max(0, Math.round((1 - (state.drowsyFrames / state.totalFrames)) * 100));
    elements.focusScoreDisplay.textContent = `${score}%`;
  }
}, 1000);

// ==============================================================================
// EVENT LISTENERS & CONTROLS
// ==============================================================================
elements.startCameraBtn.addEventListener('click', startCamera);

elements.toggleMonitorBtn.addEventListener('click', () => {
  if (state.isRunning) {
    stopCamera();
  } else {
    startCamera();
  }
});

elements.flipCameraBtn.addEventListener('click', () => {
  state.isMirrored = !state.isMirrored;
  const transformStyle = state.isMirrored ? 'scaleX(-1)' : 'scaleX(1)';
  elements.video.style.transform = transformStyle;
  elements.canvas.style.transform = transformStyle;
});

elements.toggleLandmarksBtn.addEventListener('click', () => {
  state.showWireframe = !state.showWireframe;
  elements.landmarkToggleText.textContent = state.showWireframe ? 'Hide Wireframe' : 'Show Wireframe';
});

elements.testVoiceBtn.addEventListener('click', () => {
  triggerVoiceAlert("Voice test: Study focus alerts are operational!");
});

elements.previewSoundBtn.addEventListener('click', () => {
  playAlertSound();
});

elements.resetStatsBtn.addEventListener('click', () => {
  state.sessionStartTime = Date.now();
  state.alertCount = 0;
  state.drowsyFrames = 0;
  state.totalFrames = 0;
  elements.alertCountDisplay.textContent = '0';
  elements.focusScoreDisplay.textContent = '100%';
  elements.sessionTimeDisplay.textContent = '00:00';
});

// Slider Inputs
elements.earThresholdSlider.addEventListener('input', (e) => {
  const val = parseFloat(e.target.value);
  state.earThreshold = val;
  elements.earThresholdVal.textContent = val.toFixed(2);
  elements.thresholdLabel.textContent = `Threshold: ${val.toFixed(2)}`;
  
  // Position the marker on the gauge
  const percent = (val / 0.50) * 100;
  elements.earThresholdMarker.style.left = `${percent}%`;
});

elements.sleepLimitSlider.addEventListener('input', (e) => {
  const val = parseFloat(e.target.value);
  state.sleepLimitSeconds = val;
  elements.sleepLimitVal.textContent = `${val.toFixed(1)}s`;
  updateSleepProgress(state.elapsedSleepTime);
});

elements.voiceAlertToggle.addEventListener('change', (e) => {
  state.voiceAlertEnabled = e.target.checked;
});

elements.audioBeepToggle.addEventListener('change', (e) => {
  state.audioBeepEnabled = e.target.checked;
});

// Initial threshold marker position
const initialMarkerPercent = (state.earThreshold / 0.50) * 100;
elements.earThresholdMarker.style.left = `${initialMarkerPercent}%`;
