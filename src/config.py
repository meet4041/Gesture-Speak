"""Shared application configuration and project paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MODEL_PATH = ASSETS_DIR / "gesture_recognizer.task"
SCREENSHOTS_DIR = PROJECT_ROOT / "outputs" / "screenshots"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

CAMERA_INDEX = 0
MAX_HANDS = 1
MIN_CONFIDENCE = 0.50
STABILITY_FRAMES = 6
HISTORY_LIMIT = 10

