"""OpenCV drawing helpers for GestureSpeak's dark, panel-based interface."""
from __future__ import annotations

import cv2
import numpy as np

from .gesture_mapper import display_name

BG = (22, 28, 38)
PANEL = (34, 44, 58)
TEAL = (220, 190, 30)
WHITE = (245, 247, 250)
MUTED = (175, 185, 198)
GREEN = (74, 190, 104)
YELLOW = (48, 190, 235)
RED = (75, 75, 230)


def _box(frame: np.ndarray, left: int, top: int, right: int, bottom: int, color: tuple[int, int, int] = PANEL) -> None:
    cv2.rectangle(frame, (left, top), (right, bottom), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (right, bottom), (69, 82, 99), 1, cv2.LINE_AA)


def _text(frame: np.ndarray, text: str, point: tuple[int, int], size: float, color=WHITE, thickness: int = 1) -> None:
    cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, landmarks: list | None) -> None:
    """Draw MediaPipe's 21-point hand skeleton on the camera frame."""
    if not landmarks:
        return
    height, width = frame.shape[:2]
    points = [(int(item.x * width), int(item.y * height)) for item in landmarks]
    connections = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20))
    for start, end in connections:
        cv2.line(frame, points[start], points[end], TEAL, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, WHITE, -1, cv2.LINE_AA)


def draw_interface(frame: np.ndarray, gesture: str | None, confidence: float, handedness: str | None,
                   history: list[tuple[str, str]], message: list[str], fps: float, show_history: bool) -> np.ndarray:
    """Return a polished display canvas containing video and application overlays."""
    height, width = frame.shape[:2]
    header, footer, side = 112, 74, 278 if show_history else 0
    canvas = np.full((height + header + footer, width + side, 3), BG, dtype=np.uint8)
    canvas[header:header + height, :width] = frame
    label = display_name(gesture)
    status = GREEN if confidence >= .75 and gesture else YELLOW if confidence >= .50 and gesture else RED
    _text(canvas, "GestureSpeak", (24, 42), 1.0, TEAL, 2)
    _text(canvas, "Real-time hand gesture recognition with a pretrained MediaPipe ML model", (25, 70), .48, MUTED)
    _box(canvas, width - 112, 17, width - 18, 50, (49, 61, 77))
    _text(canvas, f"{fps:4.1f} FPS", (width - 101, 40), .47, WHITE, 1)
    _box(canvas, 20, header + 16, min(340, width - 18), header + 94, PANEL)
    _text(canvas, label, (35, header + 48), .78, WHITE, 2)
    _text(canvas, f"Confidence: {confidence * 100:.0f}%", (35, header + 76), .48, status, 1)
    _box(canvas, width - 174, header + 16, width - 18, header + 94, status)
    _text(canvas, "STATUS", (width - 158, header + 45), .43, (20, 25, 32), 1)
    _text(canvas, "CONFIDENT" if status == GREEN else "CHECK" if status == YELLOW else "NO MATCH", (width - 158, header + 73), .52, (20, 25, 32), 2)
    if handedness:
        _box(canvas, 20, header + height - 52, 150, header + height - 16)
        _text(canvas, f"Hand: {handedness}", (31, header + height - 28), .46, WHITE)
    _box(canvas, 18, header + height + 13, width - 18, header + height + 59)
    built = "  ".join(message) if message else "Show a stable gesture to build a message"
    _text(canvas, "MESSAGE", (30, header + height + 33), .38, TEAL, 1)
    _text(canvas, built[:90], (116, header + height + 34), .48, WHITE if message else MUTED, 1)
    _text(canvas, "q Quit   c Clear message   h History   s Screenshot   r Reset history", (24, header + height + footer - 18), .42, MUTED)
    if show_history:
        x = width + 14
        _box(canvas, x, header, width + side - 14, header + height)
        _text(canvas, "STABLE HISTORY", (x + 14, header + 30), .48, TEAL, 1)
        if history:
            for index, (timestamp, item) in enumerate(history[:10]):
                y = header + 61 + index * 38
                _text(canvas, timestamp, (x + 14, y), .38, MUTED)
                _text(canvas, display_name(item), (x + 83, y), .42, WHITE)
        else:
            _text(canvas, "No stable gestures yet", (x + 14, header + 63), .40, MUTED)
    return canvas

