# Automated Real-Time Study Focus Monitor

An intelligent real-time computer vision application designed to maintain study focus and prevent drowsiness. It tracks facial landmarks using MediaPipe Face Mesh, computes the Eye-Aspect-Ratio (EAR), and provides non-blocking multi-modal alerts (offline voice synthesis + visual HUD overlays).

---

## Key Features

1. **Precision Eye-Aspect-Ratio (EAR) Tracking**:
   - Uses MediaPipe Face Mesh (`refine_landmarks=True`) with 468/478 iris and eyelid landmarks.
   - Calculates continuous EAR for both left and right eyes to determine eyelid aperture.
2. **Temporal Drowsiness Logic**:
   - Configurable `EAR_THRESHOLD` (default: `0.20`).
   - Temporal closed-eye timer `CLOSED_TIME_LIMIT` (default: `3.0s`).
3. **Non-Blocking TTS Audio Alerts**:
   - Threaded background worker using `pyttsx3` ensuring zero frame drop or stuttering in OpenCV.
   - Cooldown timer to prevent audio spamming or overlapping voice playback.
4. **Visual Warning & Fallback (Face Missing / Covered)**:
   - Immediate red banner alert if user leaves the frame, covers face, or turns away.
5. **Modern Heads-Up Display (HUD)**:
   - Semi-transparent glassmorphism telemetry dashboard.
   - Real-time EAR metric gauges, live sleep timer bar, FPS counter, and eye contour highlights.
6. **Graceful Termination**:
   - Exit cleanly via `'q'` key or by closing the window.

---

## Installation

1. **Clone or navigate to the workspace directory**:
   ```bash
   cd "/Users/kamran/eye monitor"
   ```

2. **(Optional) Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the study monitor:
```bash
python3 study_monitor.py
```

### Key Controls
- **`q`**: Exit the application cleanly.

---

## Customization

You can customize threshold values in [study_monitor.py](file:///Users/kamran/eye%20monitor/study_monitor.py) via `MonitorConfig`:

```python
config = MonitorConfig(
    CAMERA_INDEX=0,            # Webcam index
    EAR_THRESHOLD=0.20,        # Increase (e.g. 0.22) for earlier detection
    CLOSED_TIME_LIMIT=3.0,     # Seconds eyes can remain closed before alarm
    SPEECH_COOLDOWN=4.0,       # Minimum seconds between repeated voice alerts
    ALERT_VOICE_MESSAGE="Wake up and focus on studying!"
)
```

---

## Mathematical Formulation

The Eye Aspect Ratio (EAR) is defined as:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

Where:
- $p_1, p_4$ represent the horizontal eye corner landmarks.
- $(p_2, p_6)$ and $(p_3, p_5)$ represent vertical eyelid landmark pairs.
