#!/usr/bin/env python3
"""
Real-Time Study Focus & Drowsiness Monitor
------------------------------------------
An AI-powered computer vision system that monitors user alertness during study sessions.
Uses MediaPipe Face Mesh / Face Landmarker for sub-millimeter eye-landmark tracking,
calculates Eye-Aspect-Ratio (EAR), and executes non-blocking audio alerts (pyttsx3)
and on-screen HUD warnings.

Author: Senior Computer Vision & AI Engineer
"""

import os
import time
import queue
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
@dataclass(frozen=True)
class MonitorConfig:
    """Configuration parameters for the focus monitor."""
    # Video Capture
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720

    # Detection & EAR Thresholds
    EAR_THRESHOLD: float = 0.20          # Threshold below which eyes are considered closed
    CLOSED_TIME_LIMIT: float = 3.0       # Duration (seconds) of closed eyes before triggering alert
    SPEECH_COOLDOWN: float = 4.0         # Minimum seconds between repeated voice alerts
    
    # Model Settings
    MODEL_PATH: str = "face_landmarker.task"
    MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5

    # Alert Messages
    ALERT_VOICE_MESSAGE: str = "Wake up and focus on studying!"
    WARNING_FACE_MISSING: str = "WARNING: Face Covered or Not Detected!"


# MediaPipe Face Mesh Landmark Indices for Eyes
# Left Eye indices: [p1, p2, p3, p4, p5, p6]
# p1 (outer corner), p4 (inner corner)
# (p2, p6) and (p3, p5) are top/bottom vertical landmark pairs
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Right Eye indices: [p1, p2, p3, p4, p5, p6]
# p1 (inner corner), p4 (outer corner)
# (p2, p6) and (p3, p5) are top/bottom vertical landmark pairs
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]


# ==============================================================================
# MODEL DOWNLOAD HELPER
# ==============================================================================
def ensure_model_file(model_path: str, model_url: str):
    """Ensures that the MediaPipe Face Landmarker model asset exists locally."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 100000:
        return

    print(f"[INFO] Downloading Face Landmarker model asset to '{model_path}'...")
    try:
        # Use curl which handles macOS TLS certificates natively
        subprocess.run(["curl", "-L", "-o", model_path, model_url], check=True)
        print("[INFO] Model downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download model asset using curl: {e}")
        raise RuntimeError(f"Could not retrieve {model_path}. Please download from {model_url}")


# ==============================================================================
# THREADED TEXT-TO-SPEECH (TTS) MANAGER
# ==============================================================================
class VoiceAlertManager:
    """
    Manages non-blocking text-to-speech alerts using a dedicated background thread.
    Prevents OpenCV main capture loop stutter and eliminates audio queue congestion.
    """
    def __init__(self, speech_cooldown: float = 4.0):
        self.speech_cooldown = speech_cooldown
        self.alert_queue: queue.Queue = queue.Queue(maxsize=1)
        self.is_running = True
        self.last_speech_time = 0.0
        self.lock = threading.Lock()

        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        """Background worker thread running the pyttsx3 speech loop."""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)    # Natural speech speed
            engine.setProperty('volume', 1.0)  # Max volume
        except Exception as e:
            print(f"[ERROR] Failed to initialize TTS engine: {e}")
            return

        while self.is_running:
            try:
                # Wait for next alert message with a short timeout to allow clean shutdown
                message = self.alert_queue.get(timeout=0.2)
                if message is None:
                    break
                
                engine.say(message)
                engine.runAndWait()
                self.alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Error during TTS playback: {e}")

        try:
            engine.stop()
        except Exception:
            pass

    def trigger_alert(self, message: str) -> bool:
        """
        Submits an alert request if the cooldown period has elapsed.
        Returns True if the alert was scheduled, False otherwise.
        """
        current_time = time.time()
        with self.lock:
            if current_time - self.last_speech_time < self.speech_cooldown:
                return False
            
            # Put message in queue if space is available (non-blocking)
            try:
                self.alert_queue.put_nowait(message)
                self.last_speech_time = current_time
                return True
            except queue.Full:
                return False

    def stop(self):
        """Gracefully terminates the background TTS worker."""
        self.is_running = False
        try:
            self.alert_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)


# ==============================================================================
# COMPUTER VISION & EAR COMPUTATION FUNCTIONS
# ==============================================================================
def calculate_euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Calculates the standard Euclidean distance between two 2D points."""
    return float(np.linalg.norm(point_a - point_b))


def calculate_ear(eye_points: List[np.ndarray]) -> float:
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye given 6 contour points.
    
    Formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    """
    if len(eye_points) != 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_points

    # Vertical landmark distances
    vertical_1 = calculate_euclidean_distance(p2, p6)
    vertical_2 = calculate_euclidean_distance(p3, p5)

    # Horizontal landmark distance
    horizontal = calculate_euclidean_distance(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def extract_eye_coordinates(
    landmarks,
    indices: List[int],
    frame_width: int,
    frame_height: int
) -> List[np.ndarray]:
    """
    Extracts pixel coordinates for specified landmark indices from MediaPipe Face Landmarker.
    Supports both list of landmarks (Tasks API) and NormalizedLandmarkList (legacy).
    """
    coords = []
    landmark_list = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    for idx in indices:
        lm = landmark_list[idx]
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        coords.append(np.array([x, y], dtype=np.float32))
    return coords


# ==============================================================================
# UI & HEADS-UP DISPLAY (HUD) RENDERER
# ==============================================================================
class HUDRenderer:
    """Renders real-time telemetry, warning banners, and visual indicators on video frame."""
    
    @staticmethod
    def draw_eye_contours(frame: np.ndarray, eye_coords: List[np.ndarray], color: Tuple[int, int, int]):
        """Draws visual contour polygons and key landmark points around the eye."""
        pts = np.array([[int(p[0]), int(p[1])] for p in eye_coords], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        for point in pts:
            cv2.circle(frame, tuple(point), radius=2, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    @staticmethod
    def render_overlay_box(
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        bg_color: Tuple[int, int, int] = (20, 24, 33),
        alpha: float = 0.75
    ):
        """Draws a sleek semi-transparent background box for telemetry."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (80, 90, 110), 1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @classmethod
    def render_dashboard(
        cls,
        frame: np.ndarray,
        fps: float,
        left_ear: float,
        right_ear: float,
        avg_ear: float,
        ear_threshold: float,
        elapsed_sleep_time: float,
        sleep_limit: float,
        is_alert_active: bool
    ):
        """Renders comprehensive study focus diagnostics on the frame."""
        # Top-left Telemetry Card
        cls.render_overlay_box(frame, x=20, y=20, w=340, h=190, alpha=0.8)

        # Title
        cv2.putText(frame, "STUDY FOCUS MONITOR", (35, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 215, 0), 1, cv2.LINE_AA)
        
        # Status calculation
        if is_alert_active:
            status_text = "ALERT: DROWSY / SLEEPING"
            status_color = (0, 0, 255)       # Red
        elif elapsed_sleep_time > 0:
            status_text = "WARNING: EYES CLOSING"
            status_color = (0, 165, 255)     # Amber/Orange
        else:
            status_text = "STATUS: FOCUSED & ALERT"
            status_color = (0, 255, 127)     # Green

        # Telemetry metrics
        cv2.putText(frame, f"{status_text}", (35, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, status_color, 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"Average EAR: {avg_ear:.3f} (Threshold: {ear_threshold:.2f})", (35, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        
        cv2.putText(frame, f"L-EAR: {left_ear:.3f} | R-EAR: {right_ear:.3f}", (35, 133),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        
        # Elapsed Sleep Timer & Progress Bar
        sleep_timer_text = f"Sleep Timer: {elapsed_sleep_time:.1f}s / {sleep_limit:.1f}s"
        cv2.putText(frame, sleep_timer_text, (35, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Sleep timer bar
        bar_x, bar_y, bar_w, bar_h = 35, 175, 310, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
        fill_ratio = min(1.0, elapsed_sleep_time / max(0.1, sleep_limit))
        fill_w = int(bar_w * fill_ratio)
        bar_color = (0, 0, 255) if fill_ratio >= 1.0 else ((0, 165, 255) if fill_ratio > 0.5 else (0, 255, 127))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

        # Bottom info card (FPS & Instructions)
        h, w, _ = frame.shape
        cls.render_overlay_box(frame, x=20, y=h - 55, w=280, h=40, alpha=0.75)
        cv2.putText(frame, f"FPS: {fps:.1f} | Press 'q' to Exit", (35, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # Full-screen Alert Flash / Banner if alert active
        if is_alert_active:
            banner_y = int(h * 0.4)
            cls.render_overlay_box(frame, x=int(w * 0.15), y=banner_y, w=int(w * 0.7), h=90,
                                   bg_color=(0, 0, 180), alpha=0.85)
            cv2.putText(frame, "WAKE UP! FOCUS ON STUDYING!", (int(w * 0.15) + 30, banner_y + 55),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    @classmethod
    def render_face_missing_warning(cls, frame: np.ndarray, fps: float, message: str):
        """Renders prominent red warning overlay when user covers face or leaves frame."""
        h, w, _ = frame.shape
        
        # Red warning center banner
        banner_h = 100
        banner_y = (h - banner_h) // 2
        cls.render_overlay_box(frame, x=int(w * 0.1), y=banner_y, w=int(w * 0.8), h=banner_h,
                               bg_color=(0, 0, 200), alpha=0.85)
        
        cv2.putText(frame, message, (int(w * 0.1) + 40, banner_y + 60),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Telemetry Card indicating status
        cls.render_overlay_box(frame, x=20, y=20, w=340, h=90, alpha=0.8)
        cv2.putText(frame, "STUDY FOCUS MONITOR", (35, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 215, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "STATUS: FACE NOT DETECTED", (35, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)

        # Bottom info card (FPS & Instructions)
        cls.render_overlay_box(frame, x=20, y=h - 55, w=280, h=40, alpha=0.75)
        cv2.putText(frame, f"FPS: {fps:.1f} | Press 'q' to Exit", (35, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)


# ==============================================================================
# MAIN APPLICATION ENGINE
# ==============================================================================
class StudyFocusMonitor:
    """Main application coordinator managing video capture, inference, timing, and UI."""
    
    def __init__(self, config: MonitorConfig = MonitorConfig()):
        self.config = config
        self.voice_manager = VoiceAlertManager(speech_cooldown=config.SPEECH_COOLDOWN)
        
        # State Tracking
        self.closed_start_time: Optional[float] = None
        self.elapsed_sleep_time: float = 0.0
        self.is_alert_active: bool = False
        
        # FPS Calculation
        self.prev_frame_time: float = time.time()
        self.fps: float = 0.0

    def run(self):
        """Starts video capture stream and executes real-time processing loop."""
        ensure_model_file(self.config.MODEL_PATH, self.config.MODEL_URL)
        
        print("[INFO] Initializing MediaPipe Face Landmarker...")
        base_options = python.BaseOptions(
            model_asset_path=self.config.MODEL_PATH,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=self.config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        print("[INFO] Opening Video Capture...")
        cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)

        if not cap.isOpened():
            print(f"\n[ERROR] Could not open camera with index {self.config.CAMERA_INDEX}.")
            print("[TIP] If you are on macOS:")
            print("      1. Ensure Camera permissions are granted: System Settings -> Privacy & Security -> Camera")
            print("      2. Toggle ON permissions for your Terminal or Antigravity IDE.")
            print("      3. Run `python3 study_monitor.py` in your terminal.\n")
            self.voice_manager.stop()
            landmarker.close()
            return

        print("\n" + "=" * 60)
        print("  STUDY FOCUS MONITOR ACTIVE")
        print(f"  - EAR Threshold      : {self.config.EAR_THRESHOLD}")
        print(f"  - Closed Time Limit  : {self.config.CLOSED_TIME_LIMIT}s")
        print("  - Press 'q' in the video window to quit.")
        print("=" * 60 + "\n")

        start_timestamp_ms = int(time.time() * 1000)

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("[WARNING] Empty frame received from webcam stream. Retrying...")
                    time.sleep(0.01)
                    continue

                # Mirror frame for natural intuitive user interaction
                frame = cv2.flip(frame, 1)
                frame_h, frame_w, _ = frame.shape

                # Calculate FPS
                curr_time = time.time()
                dt = curr_time - self.prev_frame_time
                self.prev_frame_time = curr_time
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)

                # Process frame with MediaPipe Face Landmarker
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                frame_timestamp_ms = int(curr_time * 1000) - start_timestamp_ms
                detection_result = landmarker.detect_for_video(mp_image, max(1, frame_timestamp_ms))

                # ----------------------------------------------------------
                # SCENARIO A: Face Missing / Covered / Out of Frame
                # ----------------------------------------------------------
                if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
                    self.closed_start_time = None
                    self.elapsed_sleep_time = 0.0
                    self.is_alert_active = False
                    
                    HUDRenderer.render_face_missing_warning(
                        frame=frame,
                        fps=self.fps,
                        message=self.config.WARNING_FACE_MISSING
                    )
                
                # ----------------------------------------------------------
                # SCENARIO B: Face Detected - Compute EAR & Temporal Tracker
                # ----------------------------------------------------------
                else:
                    face_landmarks = detection_result.face_landmarks[0]

                    # Extract Left and Right Eye 2D coordinates
                    left_coords = extract_eye_coordinates(
                        face_landmarks, LEFT_EYE_LANDMARKS, frame_w, frame_h
                    )
                    right_coords = extract_eye_coordinates(
                        face_landmarks, RIGHT_EYE_LANDMARKS, frame_w, frame_h
                    )

                    # Calculate EAR metrics
                    left_ear = calculate_ear(left_coords)
                    right_ear = calculate_ear(right_coords)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Evaluate Eye Status against Threshold
                    if avg_ear < self.config.EAR_THRESHOLD:
                        if self.closed_start_time is None:
                            self.closed_start_time = time.time()
                        
                        self.elapsed_sleep_time = time.time() - self.closed_start_time

                        # Trigger voice alert if closed time limit exceeded
                        if self.elapsed_sleep_time >= self.config.CLOSED_TIME_LIMIT:
                            self.is_alert_active = True
                            self.voice_manager.trigger_alert(self.config.ALERT_VOICE_MESSAGE)
                        else:
                            self.is_alert_active = False
                    else:
                        # Eyes are open - reset sleep tracking
                        self.closed_start_time = None
                        self.elapsed_sleep_time = 0.0
                        self.is_alert_active = False

                    # Draw Eye Contours
                    contour_color = (0, 0, 255) if self.is_alert_active else (
                        (0, 165, 255) if self.elapsed_sleep_time > 0 else (0, 255, 127)
                    )
                    HUDRenderer.draw_eye_contours(frame, left_coords, contour_color)
                    HUDRenderer.draw_eye_contours(frame, right_coords, contour_color)

                    # Render Telemetry HUD
                    HUDRenderer.render_dashboard(
                        frame=frame,
                        fps=self.fps,
                        left_ear=left_ear,
                        right_ear=right_ear,
                        avg_ear=avg_ear,
                        ear_threshold=self.config.EAR_THRESHOLD,
                        elapsed_sleep_time=self.elapsed_sleep_time,
                        sleep_limit=self.config.CLOSED_TIME_LIMIT,
                        is_alert_active=self.is_alert_active
                    )

                # Display Live Video Feed
                cv2.imshow("Automated Real-Time Study Focus Monitor", frame)

                # Handle User Keystrokes (q: Quit)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or cv2.getWindowProperty("Automated Real-Time Study Focus Monitor", cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Exit signal received. Closing monitor...")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt detected. Exiting...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            self.voice_manager.stop()
            print("[INFO] Cleanup complete. Focus monitor terminated safely.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    config = MonitorConfig(
        EAR_THRESHOLD=0.20,
        CLOSED_TIME_LIMIT=3.0,
        SPEECH_COOLDOWN=4.0
    )
    monitor = StudyFocusMonitor(config=config)
    monitor.run()
