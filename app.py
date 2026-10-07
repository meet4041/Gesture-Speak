"""GestureSpeak desktop application entry point."""
from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np

from src.config import CAMERA_INDEX, HISTORY_LIMIT, MIN_CONFIDENCE, SCREENSHOTS_DIR, STABILITY_FRAMES
from src.gesture_mapper import gesture_to_phrase
from src.gesture_recognizer import GestureRecognizer, Prediction
from src.model_manager import ensure_model
from src.stability_filter import GestureHistory, StabilityFilter
from src.ui import draw_interface, draw_landmarks


WINDOW_NAME = "GestureSpeak"
WINDOW_BACKGROUND = (32, 18, 11)
HISTORY_ROWS = 8


def fit_to_window(canvas: np.ndarray, window_width: int, window_height: int) -> np.ndarray:
    """Scale the dashboard without stretching and fill unused space with its background."""
    if window_width <= 1 or window_height <= 1:
        return canvas
    canvas_height, canvas_width = canvas.shape[:2]
    scale = min(window_width / canvas_width, window_height / canvas_height)
    scaled_width = max(1, int(canvas_width * scale))
    scaled_height = max(1, int(canvas_height * scale))
    resized = cv2.resize(canvas, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    fitted = np.full((window_height, window_width, 3), WINDOW_BACKGROUND, dtype=np.uint8)
    left = (window_width - scaled_width) // 2
    top = (window_height - scaled_height) // 2
    fitted[top:top + scaled_height, left:left + scaled_width] = resized
    return fitted


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
    show_history, start, history_offset = True, time.perf_counter(), 0
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def scroll_history(direction: int) -> None:
        """Move through the bounded history list without letting it leave its panel."""
        nonlocal history_offset
        maximum_offset = max(0, len(history.entries) - HISTORY_ROWS)
        history_offset = max(0, min(history_offset + direction, maximum_offset))

    def on_mouse(event: int, _x: int, _y: int, flags: int, _param: object) -> None:
        """Scroll older and newer history items with the mouse wheel."""
        if event == cv2.EVENT_MOUSEWHEEL:
            # OpenCV stores the wheel delta in the signed high word of flags.
            # This works on builds that do not expose cv2.getMouseWheelDelta.
            wheel_delta = (flags >> 16) & 0xFFFF
            if wheel_delta >= 0x8000:
                wheel_delta -= 0x10000
            scroll_history(-1 if wheel_delta > 0 else 1)

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
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
                # Keep the newest entry visible once the chronological list fills.
                history_offset = max(0, len(history.entries) - HISTORY_ROWS)
            history_offset = min(history_offset, max(0, len(history.entries) - HISTORY_ROWS))
            canvas = draw_interface(
                frame, prediction.gesture, prediction.confidence, prediction.handedness,
                list(history.entries), message, show_history, history_offset=history_offset,
            )
            _, _, window_width, window_height = cv2.getWindowImageRect(WINDOW_NAME)
            cv2.imshow(WINDOW_NAME, fit_to_window(canvas, window_width, window_height))
            key = cv2.waitKeyEx(1)
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("c"):
                message.clear()
            elif key == ord("h"):
                show_history = not show_history
            elif key == ord("r"):
                history.clear(); stability.reset()
                history_offset = 0
            elif key == ord("s"):
                path = SCREENSHOTS_DIR / f"gesturespeak_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(path), canvas)
                print(f"Screenshot saved: {path}")
            elif key == 2490368:  # Up arrow: older history entries.
                scroll_history(-1)
            elif key == 2621440:  # Down arrow: newer history entries.
                scroll_history(1)
    finally:
        camera.release()
        recognizer.close()
        cv2.destroyAllWindows()
        print("GestureSpeak closed safely.")


if __name__ == "__main__":
    main()
