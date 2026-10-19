"""
Geometric features extracted from MediaPipe Face Mesh landmarks.

All functions are pure: they take a (N, 2) float array of pixel-space
landmarks and return a number, or ``None`` when the measurement is not
usable (wrong shape, NaN/inf, degenerate geometry).

Returning ``None`` - instead of 0.0 - is deliberate: a missing or invalid
measurement must never be silently treated as "eyes open" or "alert".
"""
from __future__ import annotations

import numpy as np

import config


def _as_points(points) -> np.ndarray | None:
    """Validate and convert to a finite (N, 2) float array."""
    try:
        arr = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def is_valid_landmarks(landmarks) -> bool:
    """True when ``landmarks`` is a usable full Face Mesh point set."""
    arr = _as_points(landmarks)
    return arr is not None and arr.shape[0] >= config.MIN_LANDMARKS


def ear(points) -> float | None:
    """Eye Aspect Ratio for 6 ordered eye points.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Returns ``None`` when the input is invalid or the eye width is ~0.
    """
    arr = _as_points(points)
    if arr is None or arr.shape != (6, 2):
        return None
    height = np.linalg.norm(arr[0] - arr[3])
    if height < 1e-6:
        return None
    v1 = np.linalg.norm(arr[1] - arr[5])
    v2 = np.linalg.norm(arr[2] - arr[4])
    return float((v1 + v2) / (2.0 * height))


def average_ear(landmarks) -> float | None:
    """Mean EAR of both eyes, or ``None`` if either eye is unmeasurable."""
    if not is_valid_landmarks(landmarks):
        return None
    arr = np.asarray(landmarks, dtype=float)
    left = ear(arr[config.LEFT_EYE])
    right = ear(arr[config.RIGHT_EYE])
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def mar(landmarks) -> float | None:
    """Mouth Aspect Ratio = mouth height / mouth width."""
    if not is_valid_landmarks(landmarks):
        return None
    arr = np.asarray(landmarks, dtype=float)
    height = np.linalg.norm(arr[config.MOUTH_TOP] - arr[config.MOUTH_BOTTOM])
    width = np.linalg.norm(arr[config.MOUTH_LEFT] - arr[config.MOUTH_RIGHT])
    if width < 1e-6:
        return None
    return float(height / width)


def head_down_ratio(landmarks) -> float | None:
    """How far the nose sits below the eye line, normalised by chin distance.

    0.0 = nose on the eye line, 1.0 = nose at the chin.
    """
    if not is_valid_landmarks(landmarks):
        return None
    arr = np.asarray(landmarks, dtype=float)
    eye_y = (arr[config.LEFT_EYE_CORNER][1] + arr[config.RIGHT_EYE_CORNER][1]) / 2.0
    span = arr[config.CHIN][1] - eye_y
    if abs(span) < 1e-6:
        return None
    return float((arr[config.NOSE_TIP][1] - eye_y) / span)


def head_turn_ratio(landmarks) -> float | None:
    """Horizontal nose offset from the face centre, relative to face width."""
    if not is_valid_landmarks(landmarks):
        return None
    arr = np.asarray(landmarks, dtype=float)
    x_min = float(arr[:, 0].min())
    x_max = float(arr[:, 0].max())
    width = x_max - x_min
    if width < 1e-6:
        return None
    centre = (x_min + x_max) / 2.0
    return float(abs(arr[config.NOSE_TIP][0] - centre) / width)


def face_bbox(landmarks) -> tuple[int, int, int, int] | None:
    """Integer (x1, y1, x2, y2) bounding box of the face."""
    if not is_valid_landmarks(landmarks):
        return None
    arr = np.asarray(landmarks, dtype=float)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def face_fraction(landmarks, frame_w: int, frame_h: int) -> float:
    """Face bounding-box area as a fraction of the frame (0.0 if unknown)."""
    if not is_valid_landmarks(landmarks) or frame_w <= 0 or frame_h <= 0:
        return 0.0
    arr = np.asarray(landmarks, dtype=float)
    area = ((arr[:, 0].max() - arr[:, 0].min()) *
            (arr[:, 1].max() - arr[:, 1].min()))
    return float(area) / float(frame_w * frame_h)
