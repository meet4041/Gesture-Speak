"""Display and communication mappings for MediaPipe gesture categories."""
from __future__ import annotations

GESTURE_PHRASES: dict[str, str] = {
    "Open_Palm": "HELLO",
    "Thumb_Up": "YES",
    "Thumb_Down": "NO",
    "Victory": "PEACE",
    "ILoveYou": "I LOVE YOU",
    "Closed_Fist": "STOP",
    "Pointing_Up": "ONE MOMENT",
    "OK": "OK",
    "Call_Me": "CALL ME",
    "Rock_On": "ROCK ON",
    "Finger_Gun": "YOU",
    "Three": "THREE",
    "Four": "FOUR",
}

DISPLAY_NAMES: dict[str, str] = {
    "Open_Palm": "Open Palm",
    "Closed_Fist": "Closed Fist",
    "Pointing_Up": "Pointing Up",
    "Thumb_Up": "Thumb Up",
    "Thumb_Down": "Thumb Down",
    "Victory": "Victory",
    "ILoveYou": "I Love You",
    "OK": "OK Sign",
    "Call_Me": "Call Me",
    "Rock_On": "Rock On",
    "Finger_Gun": "Finger Gun",
    "Three": "Three Fingers",
    "Four": "Four Fingers",
    "None": "No gesture",
}


def gesture_to_phrase(gesture: str) -> str | None:
    """Return the communication phrase for a recognized gesture, if any."""
    return GESTURE_PHRASES.get(gesture)


def display_name(gesture: str | None) -> str:
    """Return a readable label for a model gesture category."""
    return DISPLAY_NAMES.get(gesture or "None", gesture.replace("_", " ") if gesture else "No gesture")
