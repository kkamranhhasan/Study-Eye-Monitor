# StudyEye AI – Automated Real-Time Study Focus & Drowsiness Monitor

An intelligent, real-time Computer Vision system designed to monitor user attention, detect drowsiness, and prevent falling asleep during study or work sessions.

Runs **100% client-side in the browser** (deployable directly on **Netlify**) and also includes a standalone **Python & OpenCV desktop application**.

---

## 🌐 Deploy to Netlify (Live Web App)

The application is 100% static and runs in modern browsers with WebAssembly and WebRTC.

### Quick Netlify Deployment Steps:
1. Go to [Netlify](https://app.netlify.com/).
2. Click **Add new site** &rarr; **Import an existing project**.
3. Select **GitHub** and choose `kkamranhhasan/Study-Eye-Monitor`.
4. Netlify will automatically detect [`netlify.toml`](file:///Users/kamran/eye%20monitor/netlify.toml).
5. Click **Deploy Study-Eye-Monitor**. Your app is live instantly!

---

## 🚀 Key Web Application Features

- **In-Browser Face Mesh AI**: 468+ sub-millimeter landmark tracking via MediaPipe WebAssembly.
- **Dynamic EAR Telemetry**: Real-time Eye Aspect Ratio calculated at 30–60 FPS.
- **Voice & Sound Alerts**: Offline in-browser Text-To-Speech (*"Wake up and focus on studying!"*) using Web Speech API + synthesizer chimes.
- **Privacy First**: 100% client-side processing. No video or images ever leave your device.
- **Glassmorphism HUD**: Real-time EAR gauges, sleep timer bar, attention score, session stats, and warning overlays.

---

## 💻 Standalone Python Desktop Version

For native desktop execution with Python:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the focus monitor
python3 study_monitor.py
```

### Key Controls
- Press **`q`** in the camera window to quit gracefully.

---

## 📐 Mathematical Formulation

The Eye Aspect Ratio (EAR) is computed as:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

Where:
- $p_1, p_4$ are the outer and inner eye corners.
- $(p_2, p_6)$ and $(p_3, p_5)$ are the upper and lower eyelid landmark coordinates.
