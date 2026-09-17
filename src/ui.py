"""Drawing helpers for GestureSpeak's polished OpenCV interface."""
from __future__ import annotations

import cv2
import numpy as np

from .gesture_mapper import display_name

# OpenCV uses BGR colour values.
CANVAS = (19, 25, 35)
SURFACE = (31, 40, 54)
SURFACE_LIGHT = (43, 55, 72)
OUTLINE = (70, 86, 106)
TEAL = (223, 193, 37)
WHITE = (246, 248, 250)
MUTED = (170, 184, 199)
GREEN = (74, 190, 104)
YELLOW = (48, 190, 235)
RED = (80, 78, 229)
INK = (23, 29, 37)


def _rounded_box(
    frame: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int] = SURFACE,
    radius: int = 12,
    border: bool = True,
) -> None:
    """Draw a filled rounded card with a subtle outline."""
    radius = max(1, min(radius, (right - left) // 2, (bottom - top) // 2))
    cv2.rectangle(frame, (left + radius, top), (right - radius, bottom), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top + radius), (right, bottom - radius), color, -1, cv2.LINE_AA)
    for x, y in ((left + radius, top + radius), (right - radius, top + radius),
                 (left + radius, bottom - radius), (right - radius, bottom - radius)):
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
    if border:
        cv2.line(frame, (left + radius, top), (right - radius, top), OUTLINE, 1, cv2.LINE_AA)
        cv2.line(frame, (right, top + radius), (right, bottom - radius), OUTLINE, 1, cv2.LINE_AA)
        cv2.line(frame, (right - radius, bottom), (left + radius, bottom), OUTLINE, 1, cv2.LINE_AA)
        cv2.line(frame, (left, bottom - radius), (left, top + radius), OUTLINE, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (left + radius, top + radius), (radius, radius), 0, 180, 270, OUTLINE, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (right - radius, top + radius), (radius, radius), 0, 270, 360, OUTLINE, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (right - radius, bottom - radius), (radius, radius), 0, 0, 90, OUTLINE, 1, cv2.LINE_AA)
        cv2.ellipse(frame, (left + radius, bottom - radius), (radius, radius), 0, 90, 180, OUTLINE, 1, cv2.LINE_AA)


def _text(
    frame: np.ndarray,
    value: str,
    point: tuple[int, int],
    size: float,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, value, point, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def _status(confidence: float, gesture: str | None) -> tuple[tuple[int, int, int], str, str]:
    """Return colour and wording for the current prediction state."""
    if gesture and confidence >= 0.75:
        return GREEN, "CONFIDENT", "Stable, clear recognition"
    if gesture and confidence >= 0.50:
        return YELLOW, "CHECK", "Hold the gesture steady"
    return RED, "NO MATCH", "Show a clear hand gesture"


def draw_landmarks(frame: np.ndarray, landmarks: list | None) -> None:
    """Draw MediaPipe's 21-point hand skeleton on the live camera frame."""
    if not landmarks:
        return
    height, width = frame.shape[:2]
    points = [(int(item.x * width), int(item.y * height)) for item in landmarks]
    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
        (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    )
    for start, end in connections:
        cv2.line(frame, points[start], points[end], TEAL, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 5, INK, -1, cv2.LINE_AA)
        cv2.circle(frame, point, 3, WHITE, -1, cv2.LINE_AA)


def draw_interface(
    frame: np.ndarray,
    gesture: str | None,
    confidence: float,
    handedness: str | None,
    history: list[tuple[str, str]],
    message: list[str],
    fps: float,
    show_history: bool,
    show_onboarding: bool = False,
) -> np.ndarray:
    """Build the complete GestureSpeak dashboard around the live camera frame."""
    height, width = frame.shape[:2]
    header, footer = 104, 74
    side_width = 292 if show_history else 0
    canvas = np.full((height + header + footer, width + side_width, 3), CANVAS, dtype=np.uint8)
    canvas[header:header + height, :width] = frame

    # Header
    cv2.rectangle(canvas, (0, 0), (width + side_width, header - 1), SURFACE, -1, cv2.LINE_AA)
    cv2.line(canvas, (0, header - 1), (width + side_width, header - 1), OUTLINE, 1, cv2.LINE_AA)
    cv2.circle(canvas, (31, 34), 12, TEAL, -1, cv2.LINE_AA)
    _text(canvas, "G", (22, 41), 0.54, INK, 2)
    _text(canvas, "GestureSpeak", (53, 40), 0.96, WHITE, 2)
    _text(canvas, "REAL-TIME HAND GESTURE RECOGNITION", (55, 66), 0.39, MUTED, 1)
    _rounded_box(canvas, width - 116, 24, width - 18, 63, SURFACE_LIGHT, 10)
    _text(canvas, f"{fps:4.1f} FPS", (width - 103, 49), 0.46, WHITE, 1)

    # Camera frame and prediction card
    cv2.rectangle(canvas, (0, header), (width - 1, header + height - 1), OUTLINE, 1, cv2.LINE_AA)
    colour, status_label, status_hint = _status(confidence, gesture)
    _rounded_box(canvas, 18, header + 18, 338, header + 101, SURFACE, 12)
    _text(canvas, "CURRENT GESTURE", (33, header + 45), 0.37, MUTED, 1)
    _text(canvas, display_name(gesture), (33, header + 78), 0.75, WHITE, 2)
    _rounded_box(canvas, 18, header + 113, 201, header + 151, colour, 10, False)
    _text(canvas, f"{confidence * 100:5.1f}% confidence", (31, header + 139), 0.43, INK, 1)
    if handedness:
        _rounded_box(canvas, 213, header + 113, 338, header + 151, SURFACE, 10)
        _text(canvas, f"{handedness} hand", (226, header + 139), 0.43, WHITE, 1)

    # Clear status card at the upper right of the video.
    status_left = max(356, width - 220)
    _rounded_box(canvas, status_left, header + 18, width - 18, header + 101, colour, 12, False)
    _text(canvas, status_label, (status_left + 16, header + 50), 0.57, INK, 2)
    _text(canvas, status_hint, (status_left + 16, header + 76), 0.36, INK, 1)

    # Message builder
    message_top = header + height - 73
    _rounded_box(canvas, 18, message_top, width - 18, header + height - 18, SURFACE, 12)
    _text(canvas, "MESSAGE BUILDER", (34, message_top + 27), 0.36, TEAL, 1)
    built_message = "  |  ".join(message) if message else "Recognized phrases will appear here"
    _text(canvas, built_message[:78], (34, message_top + 56), 0.49, WHITE if message else MUTED, 1)

    # Optional right-hand history panel.
    if show_history:
        panel_left = width + 14
        panel_right = width + side_width - 14
        _rounded_box(canvas, panel_left, header + 14, panel_right, header + height - 14, SURFACE, 14)
        _text(canvas, "GESTURE HISTORY", (panel_left + 16, header + 44), 0.49, TEAL, 1)
        _text(canvas, "Latest stable recognitions", (panel_left + 16, header + 65), 0.35, MUTED, 1)
        if history:
            for index, (timestamp, item) in enumerate(history[:10]):
                top = header + 82 + index * 47
                _rounded_box(canvas, panel_left + 12, top, panel_right - 12, top + 37, SURFACE_LIGHT, 8, False)
                _text(canvas, timestamp, (panel_left + 22, top + 24), 0.34, MUTED, 1)
                _text(canvas, display_name(item)[:16], (panel_left + 86, top + 24), 0.40, WHITE, 1)
        else:
            _text(canvas, "No stable gestures yet", (panel_left + 17, header + 103), 0.40, MUTED, 1)

    # A brief, unobtrusive guide shown when the application first opens.
    if show_onboarding:
        card_width = min(560, width - 72)
        card_left = max(36, (width - card_width) // 2)
        card_top = header + max(172, height // 2 - 92)
        card_right = card_left + card_width
        card_bottom = card_top + 166
        cell_width = (card_width - 64) // 3
        first_cell = card_left + 20
        second_cell = first_cell + cell_width + 12
        third_cell = second_cell + cell_width + 12
        _rounded_box(canvas, card_left, card_top, card_right, card_bottom, SURFACE, 16)
        _text(canvas, "QUICK START", (card_left + 22, card_top + 34), 0.52, TEAL, 1)
        _text(canvas, "Hold a gesture steady until it is accepted.", (card_left + 22, card_top + 62), 0.48, WHITE, 1)
        _rounded_box(canvas, first_cell, card_top + 78, first_cell + cell_width, card_top + 117, SURFACE_LIGHT, 9, False)
        _rounded_box(canvas, second_cell, card_top + 78, second_cell + cell_width, card_top + 117, SURFACE_LIGHT, 9, False)
        _rounded_box(canvas, third_cell, card_top + 78, third_cell + cell_width, card_top + 117, SURFACE_LIGHT, 9, False)
        _text(canvas, "OPEN PALM = HELLO", (first_cell + 9, card_top + 103), 0.34, WHITE, 1)
        _text(canvas, "THUMB UP = YES", (second_cell + 9, card_top + 103), 0.34, WHITE, 1)
        _text(canvas, "VICTORY = PEACE", (third_cell + 9, card_top + 103), 0.34, WHITE, 1)
        _text(canvas, "Keep your hand visible with good lighting.", (card_left + 22, card_top + 146), 0.38, MUTED, 1)

    # Footer controls with individual key capsules.
    footer_top = header + height
    cv2.line(canvas, (0, footer_top), (width + side_width, footer_top), OUTLINE, 1, cv2.LINE_AA)
    controls = (("Q", "Quit"), ("C", "Clear"), ("H", "History"), ("S", "Screenshot"), ("R", "Reset"))
    x = 22
    for key, label in controls:
        _rounded_box(canvas, x, footer_top + 20, x + 27, footer_top + 47, SURFACE_LIGHT, 7, False)
        _text(canvas, key, (x + 8, footer_top + 39), 0.38, TEAL, 1)
        _text(canvas, label, (x + 34, footer_top + 39), 0.39, MUTED, 1)
        x += 35 + int(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.39, 1)[0][0]) + 20
    return canvas
