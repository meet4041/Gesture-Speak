"""Thin wrapper around the MediaPipe Tasks Gesture Recognizer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp

from .config import MAX_HANDS, MIN_CONFIDENCE
from .custom_gestures import recognize_custom_gesture


@dataclass(frozen=True)
class Prediction:
    """The relevant result for a single frame."""

    gesture: str | None
    confidence: float
    handedness: str | None
    landmarks: list | None


class GestureRecognizer:
    """Run synchronous VIDEO-mode hand gesture inference."""

    def __init__(self, model_path: Path) -> None:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_CONFIDENCE,
            min_hand_presence_confidence=MIN_CONFIDENCE,
            min_tracking_confidence=MIN_CONFIDENCE,
        )
        self._recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

    def recognize(self, rgb_frame, timestamp_ms: int) -> Prediction:
        """Recognize the first hand in an RGB frame."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._recognizer.recognize_for_video(image, timestamp_ms)
        if not result.gestures:
            return Prediction(None, 0.0, None, None)
        categories = result.gestures[0]
        gesture = categories[0] if categories else None
        hands = result.handedness[0] if result.handedness else []
        hand = hands[0] if hands else None
        landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
        custom_gesture = recognize_custom_gesture(landmarks, hand.category_name if hand else None)
        return Prediction(
            custom_gesture or (gesture.category_name if gesture else None),
            # Landmark rules are deterministic; do not let an unrelated or
            # absent pretrained-model category suppress a valid custom sign.
            0.95 if custom_gesture else (float(gesture.score) if gesture else 0.0),
            hand.category_name if hand else None,
            landmarks,
        )

    def close(self) -> None:
        """Release native MediaPipe resources."""
        self._recognizer.close()
