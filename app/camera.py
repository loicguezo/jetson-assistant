# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
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
        jpeg_quality: int = 80,
        camera_type: int = cv2.CAP_V4L2,
    ):
        self._device = device
        self.width = width
        self.height = height
        self._framerate = framerate
        self.jpeg_quality = jpeg_quality
        self._type = camera_type

        self._cap: cv2.VideoCapture | None = None
        self.camera_specs = None
        self.camera_K = None
        self.camera_D = None

        self._cap_lock = threading.Lock()
        self._frame_lock = threading.Lock()

        self._latest_frame: np.ndarray | None = None
        self._latest_jpeg: bytes | None = None

        self._alive = False
        self._thread: threading.Thread | None = None

    def _build_gstreamer_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc wbmode=1 ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self._framerate}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
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
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            case _:
                raise TypeError("Camera type undefined.")
        return cap

    def _sync_actual_frame_geometry(self) -> None:
        if self._cap is None:
            return
        if self._type == cv2.CAP_GSTREAMER:
            return
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width > 0:
            self.width = actual_width
        if actual_height > 0:
            self.height = actual_height

    def open(self) -> bool:
        with self._cap_lock:
            self._cap = self._init_camera()
            if self._cap is not None and self._cap.isOpened():
                self._sync_actual_frame_geometry()
                return True
            return False

    def start(self) -> bool:
        if not self.open():
            return False
        self._alive = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _capture_loop(self):
        last_success = time.monotonic()
        interval = 1.0 / self._framerate

        while self._alive:
            t_start = time.monotonic()

            with self._cap_lock:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(0.1)
                    continue
                ret, frame = self._cap.read()
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                cv2.imshow("w1", frame)

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

            jpg_bytes = self._encode_frame_bytes(frame)

            with self._frame_lock:
                self._latest_frame = frame
                self._latest_jpeg = jpg_bytes

            elapsed = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _encode_frame_bytes(self, frame: np.ndarray) -> bytes | None:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return jpg.tobytes() if ok else None

    @staticmethod
    def _base64_jpeg(jpg: bytes) -> str:
        return base64.b64encode(jpg).decode("ascii")

    def read_live(self) -> str | None:
        with self._frame_lock:
            jpg = self._latest_jpeg

        return self._base64_jpeg(jpg) if jpg is not None else None

    def read_raw_live(self, copy: bool = True) -> np.ndarray | None:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def read(self) -> np.ndarray | None:
        return self.read_raw_live()

    def get_speech_frames(self) -> list[str]:
        frame_b64 = self.read_live()
        return [frame_b64] if frame_b64 else []

    def health_check(self) -> bool:
        return self._cap is not None and self._cap.isOpened() and self._alive

    def close(self):
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
            self._latest_jpeg = None
        logging.info("Camera successfully stopped.")

    def stop(self):
        self.close()
