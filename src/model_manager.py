"""Download and locate the official MediaPipe gesture-recognizer model."""
from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path

from .config import MODEL_PATH, MODEL_URL


def ensure_model(model_path: Path = MODEL_PATH, model_url: str = MODEL_URL) -> Path:
    """Return a local model path, downloading the official model when needed.

    The download is written to a temporary sibling first so an interrupted download
    never leaves a corrupt model at the final path.
    """
    if model_path.is_file() and model_path.stat().st_size > 0:
        print(f"Using pretrained model: {model_path}")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".task.download")
    print("Pretrained gesture model is missing. Downloading it once...")
    try:
        with urllib.request.urlopen(model_url, timeout=60) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary_path.stat().st_size == 0:
            raise OSError("The downloaded file was empty.")
        temporary_path.replace(model_path)
        print("Model download complete.")
        return model_path
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the pretrained MediaPipe model. "
            f"Check your internet connection and try again. Direct URL: {model_url}"
        ) from error

