"""
System Health, Edge Analytics & Camera Obstruction Detection for SmartPark Vision.
Monitors processed FPS, CPU/GPU/memory health, and detects lens occlusions or sudden shifts.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import cv2
import psutil

class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_calc_time = time.time()
        self.fps_frame_count = 0
        
        # Camera shift / occlusion detection baseline
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.occlusion_alert = False
        self.shift_alert = False

    def update_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Calculates real-time FPS and verifies frame health (lens occlusion, signal loss).
        """
        self.frame_count += 1
        self.fps_frame_count += 1
        current_time = time.time()
        
        # Calculate smoothed FPS every 1.0 second
        if current_time - self.last_fps_calc_time >= 1.0:
            self.fps = self.fps_frame_count / (current_time - self.last_fps_calc_time)
            self.fps_frame_count = 0
            self.last_fps_calc_time = current_time

        # Check camera health
        is_healthy, anomaly_reason = self._check_camera_health(frame)

        return {
            "fps": round(self.fps, 1),
            "uptime_seconds": int(current_time - self.start_time),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "camera_healthy": is_healthy,
            "anomaly_reason": anomaly_reason
        }

    def _check_camera_health(self, frame: np.ndarray) -> (bool, Optional[str]):
        if frame is None or frame.size == 0:
            return False, "signal_loss"

        # Downsample for quick computation
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # 1. Total blackout / high glare check (mean pixel brightness & standard deviation)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        if std_brightness < 4.0:
            return False, "lens_occlusion_or_blank"
        
        if mean_brightness < 5.0:
            return False, "severe_darkness_signal_loss"

        # 2. Frame difference for sudden shift detection
        if self.prev_frame_gray is not None:
            diff = cv2.absdiff(gray, self.prev_frame_gray)
            mean_diff = np.mean(diff)
            # Extreme global delta indicates sudden camera displacement
            if mean_diff > 120.0:
                self.prev_frame_gray = gray
                return False, "sudden_camera_shift"

        self.prev_frame_gray = gray
        return True, None
