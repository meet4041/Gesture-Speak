"""Rule-based gestures derived from MediaPipe's 21 hand landmarks."""
from __future__ import annotations

from math import hypot


_TIP = (4, 8, 12, 16, 20)
_PIP = (3, 6, 10, 14, 18)


def _finger_states(landmarks: list, handedness: str | None) -> tuple[bool, bool, bool, bool, bool]:
    """Return whether thumb, index, middle, ring and pinky are extended."""
    thumb_tip, thumb_joint = landmarks[_TIP[0]], landmarks[_PIP[0]]
    thumb_extended = thumb_tip.x > thumb_joint.x if handedness == "Left" else thumb_tip.x < thumb_joint.x
    fingers = [thumb_extended]
    for tip_index, pip_index in zip(_TIP[1:], _PIP[1:]):
        fingers.append(landmarks[tip_index].y < landmarks[pip_index].y)
    return tuple(fingers)  # type: ignore[return-value]


def _near(first, second, scale: float) -> bool:
    return hypot(first.x - second.x, first.y - second.y) < scale


def recognize_custom_gesture(landmarks: list | None, handedness: str | None) -> str | None:
    """Identify six additional clear finger configurations, if present."""
    if not landmarks or len(landmarks) < 21:
        return None
    thumb, index, middle, ring, pinky = _finger_states(landmarks, handedness)
    wrist, middle_mcp = landmarks[0], landmarks[9]
    hand_scale = max(hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y), 0.03)
    if _near(landmarks[4], landmarks[8], hand_scale * 0.55) and middle and ring and pinky:
        return "OK"
    if thumb and not index and not middle and not ring and pinky:
        return "Call_Me"
    if not thumb and index and not middle and not ring and pinky:
        return "Rock_On"
    if thumb and index and not middle and not ring and not pinky:
        return "Finger_Gun"
    if not thumb and index and middle and ring and not pinky:
        return "Three"
    if not thumb and index and middle and ring and pinky:
        return "Four"
    return None
