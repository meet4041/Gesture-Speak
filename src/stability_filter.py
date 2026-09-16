"""Temporal filtering and bounded gesture history."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StabilityFilter:
    """Emit a gesture once after it has persisted for a required number of frames."""

    required_frames: int = 6
    _candidate: str | None = field(default=None, init=False)
    _count: int = field(default=0, init=False)
    _emitted: str | None = field(default=None, init=False)

    def update(self, gesture: str | None) -> str | None:
        """Process one prediction and return a newly stable gesture only once."""
        if not gesture or gesture == "None":
            self._candidate, self._count, self._emitted = None, 0, None
            return None
        if gesture != self._candidate:
            self._candidate, self._count = gesture, 1
            self._emitted = None
        else:
            self._count += 1
        if self._count >= self.required_frames and self._emitted != gesture:
            self._emitted = gesture
            return gesture
        return None

    def reset(self) -> None:
        """Clear temporal state."""
        self._candidate, self._count, self._emitted = None, 0, None


@dataclass
class GestureHistory:
    """Keep the newest timestamped stable gestures up to a fixed limit."""

    limit: int = 10
    entries: deque[tuple[str, str]] = field(default_factory=deque)

    def add(self, gesture: str, timestamp: str | None = None) -> None:
        """Add a stable gesture with a display timestamp."""
        self.entries.appendleft((timestamp or datetime.now().strftime("%H:%M:%S"), gesture))
        while len(self.entries) > self.limit:
            self.entries.pop()

    def clear(self) -> None:
        """Remove all recorded gestures."""
        self.entries.clear()

