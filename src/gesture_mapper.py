"""Display and communication mappings for MediaPipe gesture categories."""
from __future__ import annotations

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
def display_name(gesture: str | None) -> str:
    """Return a readable label for a model gesture category."""
    return DISPLAY_NAMES.get(gesture or "None", gesture.replace("_", " ") if gesture else "No gesture")
