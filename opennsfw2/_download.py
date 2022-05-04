"""
Download utilities.
"""
import os
from pathlib import Path

import gdown  # type: ignore

WEIGHTS_FILE = "open_nsfw_weights.h5"
WEIGHTS_URL = f"https://github.com/bhky/opennsfw2/releases/download/v0.1.0/{WEIGHTS_FILE}"


def _get_home_dir() -> str:
    return str(os.getenv("OPENNSFW2_HOME", default=Path.home()))


def get_default_weights_path() -> str:
    home_dir = _get_home_dir()
    return os.path.join(home_dir, f".opennsfw2/weights/{WEIGHTS_FILE}")


def download_weights_to(weights_path: str) -> None:
    download_dir = os.path.dirname(os.path.abspath(weights_path))
    os.makedirs(download_dir, exist_ok=True)
    print(f"Pre-trained weights will be downloaded.")
    gdown.download(WEIGHTS_URL, weights_path)
