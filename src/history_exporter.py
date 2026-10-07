"""CSV export support for GestureSpeak history."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .gesture_mapper import display_name


def export_history_csv(entries: Iterable[tuple[str, str]], output_dir: Path) -> Path:
    """Export newest-first history entries as a chronological CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"gesture_history_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    chronological_entries = list(reversed(list(entries)))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("serial_number", "timestamp", "gesture"))
        for serial_number, (timestamp, gesture) in enumerate(chronological_entries, start=1):
            writer.writerow((serial_number, timestamp, display_name(gesture)))
    return path
