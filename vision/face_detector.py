"""
MediaPipe Face Mesh wrapper.

Returns plain ``numpy`` landmark arrays in pixel coordinates, so the rest of
the application never depends on MediaPipe's result objects.
"""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

import config


class FaceDetector:
    """Detect up to ``max_faces`` faces and return their 468+ landmarks."""

    def __init__(self,
                 max_faces: int = 2,
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5,
                 refine_landmarks: bool = True):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        """Return one (N, 2) pixel-space array per detected face."""
        if frame_bgr is None:
            return []
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []
        faces = []
        for face in result.multi_face_landmarks:
            faces.append(np.array(
                [(p.x * width, p.y * height) for p in face.landmark],
                dtype=float,
            ))
        return faces

    def close(self) -> None:
        try:
            self.mesh.close()
        except Exception:
            pass

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def select_driver(faces: list[np.ndarray],
                  frame_w: int,
                  frame_h: int,
                  anchor: float | None = None,
                  side_bias: float | None = None) -> int | None:
    """Return the index of the driver's face (largest, biased to the seat).

    Scoring is done in normalised coordinates so the result is independent
    of the frame resolution. Returns ``None`` when no faces were detected.
    """
    if not faces:
        return None
    anchor = config.DRIVER_ANCHOR if anchor is None else anchor
    side_bias = config.SIDE_BIAS if side_bias is None else side_bias
    if frame_w <= 0 or frame_h <= 0:
        return 0

    def score(face: np.ndarray) -> float:
        xs = face[:, 0] / float(frame_w)
        ys = face[:, 1] / float(frame_h)
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        centre = (xs.max() + xs.min()) / 2.0
        return float(area - side_bias * abs(centre - anchor))

    return int(max(range(len(faces)), key=lambda i: score(faces[i])))
