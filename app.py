"""GestureSpeak desktop application entry point."""
from __future__ import annotations

import time
from datetime import datetime

import cv2

from src.config import CAMERA_INDEX, HISTORY_LIMIT, MIN_CONFIDENCE, SCREENSHOTS_DIR, STABILITY_FRAMES
from src.gesture_mapper import gesture_to_phrase
from src.gesture_recognizer import GestureRecognizer, Prediction
from src.model_manager import ensure_model
from src.stability_filter import GestureHistory, StabilityFilter
from src.ui import draw_interface, draw_landmarks


def main() -> None:
    """Start GestureSpeak and safely release resources when it closes."""
    try:
        model_path = ensure_model()
    except RuntimeError as error:
        print(f"\n{error}")
        return
    try:
        recognizer = GestureRecognizer(model_path)
    except Exception as error:
        print(f"Could not start MediaPipe Gesture Recognizer: {error}")
        return
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        recognizer.close()
        print("Could not open the webcam. Check camera permissions or try another camera index.")
        return
    stability, history, message = StabilityFilter(STABILITY_FRAMES), GestureHistory(HISTORY_LIMIT), []
    show_history, start, previous, fps = True, time.perf_counter(), time.perf_counter(), 0.0
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("GestureSpeak is running. Press q in the video window to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam; closing safely.")
                break
            frame = cv2.flip(frame, 1)
            timestamp = max(1, int((time.perf_counter() - start) * 1000))
            prediction: Prediction = recognizer.recognize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp)
            draw_landmarks(frame, prediction.landmarks)
            stable = stability.update(
                prediction.gesture if prediction.confidence >= MIN_CONFIDENCE else None
            )
            if stable:
                history.add(stable)
                phrase = gesture_to_phrase(stable)
                if phrase:
                    message.append(phrase)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - previous, 0.001)
            previous = now
            canvas = draw_interface(frame, prediction.gesture, prediction.confidence, prediction.handedness, list(history.entries), message, fps, show_history)
            cv2.imshow("GestureSpeak", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty("GestureSpeak", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("c"):
                message.clear()
            elif key == ord("h"):
                show_history = not show_history
            elif key == ord("r"):
                history.clear(); stability.reset()
            elif key == ord("s"):
                path = SCREENSHOTS_DIR / f"gesturespeak_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(path), canvas)
                print(f"Screenshot saved: {path}")
    finally:
        camera.release()
        recognizer.close()
        cv2.destroyAllWindows()
        print("GestureSpeak closed safely.")


if __name__ == "__main__":
    main()
