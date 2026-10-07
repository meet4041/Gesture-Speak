"""Display and communication mappings for MediaPipe gesture categories."""
from __future__ import annotations

GESTURE_PHRASES: dict[str, str] = {
    "Open_Palm": "HELLO",
    "Thumb_Up": "YES",
    "Thumb_Down": "NO",
    "Victory": "PEACE",
    "Closed_Fist": "STOP",
}

GESTURE_ORDER = ("Open_Palm", "Closed_Fist", "Thumb_Up", "Thumb_Down", "Victory")
ENABLED_GESTURES = frozenset(GESTURE_ORDER)

DISPLAY_NAMES: dict[str, str] = {
    "Open_Palm": "Open Palm",
    "Closed_Fist": "Closed Fist",
    "Thumb_Up": "Thumb Up",
    "Thumb_Down": "Thumb Down",
    "Victory": "Victory",
    "None": "No gesture",
}


def gesture_to_phrase(gesture: str) -> str | None:
    """Return the communication phrase for a recognized gesture, if any."""
    return GESTURE_PHRASES.get(gesture)


def display_name(gesture: str | None) -> str:
    """Return a readable label for a model gesture category."""
    return DISPLAY_NAMES.get(gesture or "None", gesture.replace("_", " ") if gesture else "No gesture")
