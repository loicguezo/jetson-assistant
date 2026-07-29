import logging
import threading
import time

import cv2
import numpy as np

APP_CAMERA_FRAMERATE = 3.0


class Camera:
    def __init__(
        self,
        device: str | int = 0,
        width: int = 1920,
        height: int = 1080,
        framerate: int = 3,
        type: int = cv2.CAP_V4L2,
    ):
        self._device = device
        self._width = width
        self._height = height
        self._framerate = framerate
        self._cap: cv2.VideoCapture | None = None
        self._cap_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._alive = False
        self._thread: threading.Thread | None = None
        self._type = type

    def _build_gstreamer_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self._width}, height={self._height}, "
            f"format=NV12, framerate={self._framerate}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self._width}, height={self._height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink drop=true"
        )

    def _init_camera(self) -> cv2.VideoCapture:
        match self._type:
            case cv2.CAP_GSTREAMER:
                pipeline = self._build_gstreamer_pipeline()
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            case cv2.CAP_V4L2:
                cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            case _:
                raise TypeError("Camera type undefined.")
        return cap

    def open(self) -> bool:
        with self._cap_lock:
            if self._cap is not None and self._cap.isOpened():
                return True
            self._cap = self._init_camera()
            return self._cap.isOpened()

    def start(self) -> bool:
        if not self.open():
            return False
        self._alive = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        last_success = time.monotonic()

        while self._alive:
            with self._cap_lock:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(0.1)
                    continue
                ret, frame = self._cap.read()

            now = time.monotonic()

            if not ret or frame is None:
                if now - last_success > APP_CAMERA_FRAMERATE:
                    with self._cap_lock:
                        if self._cap is not None:
                            self._cap.release()
                        self._cap = self._init_camera()
                    last_success = now
                time.sleep(0.01)
                continue

            last_success = now

            with self._frame_lock:
                self._latest_frame = frame

            time.sleep(0.001)

    def read(self) -> np.ndarray | None:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def stop(self):
        self._alive = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        with self._frame_lock:
            self._latest_frame = None
        logging.info("Camera perfectly stopped.")
