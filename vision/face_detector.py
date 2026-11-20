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
                 detection_confidence: float | None = None,
                 tracking_confidence: float | None = None,
                 refine_landmarks: bool = True):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=float(
                config.DETECTION_CONFIDENCE if detection_confidence is None
                else detection_confidence),
            min_tracking_confidence=float(
                config.TRACKING_CONFIDENCE if tracking_confidence is None
                else tracking_confidence),
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


def centre_crop(frame_w: int, frame_h: int,
                factor: float | None = None) -> tuple[int, int, int, int]:
    """Return the (x, y, w, h) of a centred crop ``factor`` times smaller.

    Used as a digital zoom: when the driver is far from the camera the face is
    only a handful of pixels wide, so we crop the middle of the frame, upscale
    it back to full size and let MediaPipe try again.
    """
    zoom = max(1.0, float(config.ZOOM_FACTOR if factor is None else factor))
    width = max(1, int(round(frame_w / zoom)))
    height = max(1, int(round(frame_h / zoom)))
    width = min(width, int(frame_w))
    height = min(height, int(frame_h))
    return ((int(frame_w) - width) // 2, (int(frame_h) - height) // 2,
            width, height)


def remap_landmarks(faces: list[np.ndarray],
                    crop_x: int, crop_y: int,
                    crop_w: int, crop_h: int,
                    frame_w: int, frame_h: int) -> list[np.ndarray]:
    """Map landmarks found on an upscaled crop back to original frame pixels.

    The crop is resized to ``frame_w x frame_h`` before detection, so a point
    ``p`` in the crop frame corresponds to
    ``crop_x + p.x * crop_w / frame_w`` in the original frame.
    """
    if frame_w <= 0 or frame_h <= 0:
        return [np.array(face, dtype=float, copy=True) for face in faces]
    scale_x = float(crop_w) / float(frame_w)
    scale_y = float(crop_h) / float(frame_h)
    mapped: list[np.ndarray] = []
    for face in faces:
        points = np.array(face, dtype=float, copy=True)
        if points.size:
            points[:, 0] = points[:, 0] * scale_x + float(crop_x)
            points[:, 1] = points[:, 1] * scale_y + float(crop_y)
        mapped.append(points)
    return mapped


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
