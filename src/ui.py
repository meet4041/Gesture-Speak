"""Drawing helpers for GestureSpeak's polished OpenCV interface."""
from __future__ import annotations

import cv2
import numpy as np

from .gesture_mapper import display_name

# OpenCV uses BGR colour values. These values render as a cool navy interface,
# rather than the warm/brown palette created by RGB-ordered values.
CANVAS = (32, 18, 11)
SURFACE = (46, 28, 17)
SURFACE_LIGHT = (59, 38, 24)
LATEST_SURFACE = (70, 61, 22)
OUTLINE = (76, 54, 38)
TEAL = (238, 211, 34)
WHITE = (250, 248, 246)
MUTED = (184, 163, 148)
INK = (32, 18, 11)


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
    show_history: bool,
    history_offset: int = 0,
) -> np.ndarray:
    """Build the complete GestureSpeak dashboard around the live camera frame."""
    header, footer = 36, 0
    if not show_history:
        # Match a 16:9 full-screen canvas by cropping only the camera's top/bottom.
        # This prevents empty side bands while preserving the camera's proportions.
        source_height, source_width = frame.shape[:2]
        target_camera_height = int(source_width * 9 / 16) - header
        if 0 < target_camera_height < source_height:
            top = (source_height - target_camera_height) // 2
            frame = frame[top:top + target_camera_height, :]
    height, width = frame.shape[:2]
    side_width = 330 if show_history else 0
    canvas = np.full((height + header + footer, width + side_width, 3), CANVAS, dtype=np.uint8)
    canvas[header:header + height, :width] = frame

    # Header
    cv2.rectangle(canvas, (0, 0), (width + side_width, header - 1), SURFACE, -1, cv2.LINE_AA)
    cv2.line(canvas, (0, header - 1), (width + side_width, header - 1), OUTLINE, 1, cv2.LINE_AA)
    cv2.circle(canvas, (15, 14), 5, TEAL, -1, cv2.LINE_AA)
    _text(canvas, "G", (12, 17), 0.22, INK, 1)
    _text(canvas, "GestureSpeak", (27, 19), 0.38, WHITE, 1)

    # Compact controls stay available without consuming a large footer.
    controls = (("Q", "Quit"), ("H", "History"), ("S", "Shot"), ("R", "Reset"))
    header_right = width + side_width
    control_widths = [
        24 + int(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.24, 1)[0][0])
        for _, label in controls
    ]
    x = header_right - sum(control_widths) - 8 * (len(controls) - 1) - 16
    for (key, label), item_width in zip(controls, control_widths):
        _rounded_box(canvas, x, 8, x + 17, 27, SURFACE_LIGHT, 6, False)
        _text(canvas, key, (x + 6, 22), 0.25, TEAL, 1)
        _text(canvas, label, (x + 22, 21), 0.24, MUTED, 1)
        x += item_width + 8

    # Camera frame remains clean; recognition details appear in the history panel.
    cv2.rectangle(canvas, (0, header), (width - 1, header + height - 1), OUTLINE, 1, cv2.LINE_AA)

    # Optional right-hand history panel.
    if show_history:
        panel_left = width + 14
        panel_right = width + side_width - 14
        _rounded_box(canvas, panel_left, header + 14, panel_right, header + height - 14, SURFACE, 14)
        _text(canvas, "GESTURE HISTORY", (panel_left + 16, header + 44), 0.49, TEAL, 1)
        _text(canvas, "Oldest to newest", (panel_left + 16, header + 65), 0.35, MUTED, 1)
        if history:
            visible_rows = 8
            chronological_history = list(reversed(history))
            visible_history = chronological_history[history_offset:history_offset + visible_rows]
            for index, (timestamp, item) in enumerate(visible_history):
                top = header + 82 + index * 42
                is_latest = history_offset + index == len(chronological_history) - 1
                row_colour = LATEST_SURFACE if is_latest else SURFACE_LIGHT
                _rounded_box(canvas, panel_left + 12, top, panel_right - 12, top + 34, row_colour, 8, False)
                if is_latest:
                    cv2.line(canvas, (panel_left + 18, top + 8), (panel_left + 18, top + 26), TEAL, 2, cv2.LINE_AA)
                serial_number = history_offset + index + 1
                _text(canvas, f"{serial_number}.", (panel_left + 28, top + 22), 0.29, TEAL, 1)
                _text(canvas, timestamp, (panel_left + 50, top + 22), 0.29, MUTED, 1)
                _text(canvas, display_name(item)[:14], (panel_left + 111, top + 22), 0.35, WHITE, 2 if is_latest else 1)
            if len(history) > visible_rows:
                first = history_offset + 1
                last = min(history_offset + visible_rows, len(history))
                _text(canvas, f"{first}-{last} of {len(history)}  |  scroll: mouse wheel or up/down", (panel_left + 16, header + height - 28), 0.25, MUTED, 1)
        else:
            _text(canvas, "No stable gestures yet", (panel_left + 17, header + 103), 0.40, MUTED, 1)

    return canvas
