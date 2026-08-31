"""
Threaded Video Stream & Frame Buffer for SmartPark Vision.
Decouples RTSP/Video I/O frame decoding from YOLOv11 inference and spatial processing.
Eliminates I/O blocking and maximizes FPS throughput on Edge nodes.
"""

import time
import threading
import queue
from typing import Optional, Tuple
import cv2
import numpy as np

class ThreadedVideoStream:
    """
    High-throughput threaded frame capture stream with zero-latency buffer management.
    """
    def __init__(self, source_path_or_rtsp: str, queue_size: int = 3, is_live_stream: Optional[bool] = None):
        self.source = source_path_or_rtsp
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video stream: {self.source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Auto-detect if live RTSP / webcam vs static video file
        if is_live_stream is None:
            self.is_live = str(self.source).startswith(("rtsp://", "http://", "https://")) or str(self.source).isdigit()
        else:
            self.is_live = is_live_stream

        # Thread-safe queue
        self.frame_queue = queue.Queue(maxsize=1 if self.is_live else queue_size)
        self.stopped = False
        self.total_read_frames = 0
        
        # Background worker thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="VideoStreamWorker")
        self.thread.start()

    def _capture_loop(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break

            self.total_read_frames += 1

            if self.is_live:
                # For live streams: discard stale frame if buffer is full to ensure real-time latency
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass
            else:
                # For video files: block until consumer makes space to prevent skipping
                while not self.stopped:
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        self.cap.release()

    def read(self, timeout: float = 2.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the next decoded frame from the queue.
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            if self.stopped:
                return False, None
            return False, None

    def is_running(self) -> bool:
        return not self.stopped or not self.frame_queue.empty()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
