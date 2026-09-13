"""
Download utilities.
"""
import os
from pathlib import Path

import gdown  # type: ignore

WEIGHTS_FILE = "open_nsfw_weights.h5"
WEIGHTS_URL = f"https://github.com/bhky/opennsfw2/releases/download/v0.1.0/{WEIGHTS_FILE}"

# First bytes of any HDF5 file.
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def _get_home_dir() -> str:
    return str(os.getenv("OPENNSFW2_HOME", default=Path.home()))


def get_default_weights_path() -> str:
    home_dir = _get_home_dir()
    return os.path.join(home_dir, f".opennsfw2/weights/{WEIGHTS_FILE}")


def _is_hdf5_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE
    except OSError:
        return False


def download_weights_to(weights_path: str) -> None:
    download_dir = os.path.dirname(os.path.abspath(weights_path))
    os.makedirs(download_dir, exist_ok=True)
    print("Pre-trained weights will be downloaded.")
    gdown.download(WEIGHTS_URL, weights_path)

    # A proxy or an error page yields a small non-HDF5 file that gdown reports as
    # a success. Such a file must not be left behind: it looks cached, so it is
    # never re-downloaded and every later load fails.
    if not _is_hdf5_file(weights_path):
        try:
            os.remove(weights_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Downloaded weights file is not a valid HDF5 file, "
            f"the download from {WEIGHTS_URL} likely failed or was intercepted."
        )
