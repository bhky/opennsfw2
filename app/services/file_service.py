"""
File service for handling input data (URLs and base64).
"""
import base64
import binascii
import io
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from typing import Generator, Optional, Union
from urllib.parse import urlparse
import requests

from PIL import Image

import opennsfw2
from ..pydantic_models import InputType, InputData
from ..utils.exceptions import InvalidInputError, DownloadError

# Only these schemes are fetched; `requests` would otherwise try odd ones.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

# Total wall-clock budget for one download. The `requests` timeout only bounds
# each individual socket read, so a slow-drip origin never trips it.
DOWNLOAD_DEADLINE_SECONDS = int(
    os.getenv("OPENNSFW2_DOWNLOAD_DEADLINE_SECONDS", "120")
)

_DATA_URI_PREFIX_PATTERN = re.compile(r"^data:[^;,]*(;[^;,]+)*;base64,", re.IGNORECASE)


class FileService:
    """Service for handling file operations and input processing."""

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if a URL is valid and uses an allowed scheme."""
        try:
            result = urlparse(url)
            return bool(result.netloc) and result.scheme in _ALLOWED_URL_SCHEMES
        except Exception:
            return False

    @staticmethod
    def download_from_url(
            url: str,
            timeout: int = 30,
            deadline_seconds: Optional[int] = None
    ) -> bytes:
        """
        Download file content from URL.

        Args:
            url: URL to download from.
            timeout: Per-read socket timeout in seconds.
            deadline_seconds: Total wall-clock budget for the whole download.
                Defaults to `DOWNLOAD_DEADLINE_SECONDS`.

        Returns:
            File content as bytes.

        Raises:
            DownloadError: If download fails or exceeds the deadline.
        """
        if not FileService.is_valid_url(url):
            raise InvalidInputError(f"Invalid URL: {url}")

        if deadline_seconds is None:
            deadline_seconds = DOWNLOAD_DEADLINE_SECONDS

        headers = {
            "User-Agent": f"OpenNSFW2-API/{opennsfw2.__version__}"
        }
        expired = threading.Event()
        try:
            with requests.get(
                    url, timeout=timeout, headers=headers, stream=True
            ) as response:
                response.raise_for_status()

                # The deadline must be enforced by shutting the socket down.
                # `iter_content` blocks until a whole chunk arrives, so a check
                # inside the loop never runs against a slow-drip origin, and
                # `Response.close()` only returns the connection to the pool.
                def on_deadline() -> None:
                    expired.set()
                    try:
                        response.raw.shutdown()
                    except Exception:  # pylint: disable=broad-except
                        response.close()

                watchdog = threading.Timer(deadline_seconds, on_deadline)
                watchdog.daemon = True
                watchdog.start()
                try:
                    # Collect chunks and join once. Repeated `content += chunk`
                    # is quadratic and makes a large download look like a hang.
                    chunks = []
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if expired.is_set():
                            break
                        chunks.append(chunk)
                except Exception as e:
                    if expired.is_set():
                        raise DownloadError(
                            f"Download exceeded the {deadline_seconds}s deadline: {url}"
                        ) from e
                    raise
                finally:
                    watchdog.cancel()

                if expired.is_set():
                    raise DownloadError(
                        f"Download exceeded the {deadline_seconds}s deadline: {url}"
                    )

                return b"".join(chunks)

        except requests.RequestException as e:
            if expired.is_set():
                raise DownloadError(
                    f"Download exceeded the {deadline_seconds}s deadline: {url}"
                ) from e
            raise DownloadError(f"Failed to download from URL: {e}") from e

    @staticmethod
    def decode_base64(data: str) -> bytes:
        """
        Decode base64 string to bytes.

        A `data:` URI prefix and any whitespace are accepted, so that output of
        the `base64` CLI (wrapped at 76 columns) and browser data URIs work.

        Args:
            data: Base64 encoded string.

        Returns:
            Decoded bytes.

        Raises:
            InvalidInputError: If base64 is invalid.
        """
        cleaned = _DATA_URI_PREFIX_PATTERN.sub("", data.strip())
        cleaned = "".join(cleaned.split())

        try:
            return base64.b64decode(cleaned, validate=True)
        except (binascii.Error, ValueError) as e:
            raise InvalidInputError(f"Invalid base64 data: {e}") from e

    @staticmethod
    @contextmanager
    def get_temp_file(content: bytes, suffix: str = "") -> Generator[str, None, None]:
        """
        Create a temporary file with the given content.

        Args:
            content: File content as bytes.
            suffix: File suffix/extension.

        Yields:
            Path to temporary file.
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            yield tmp_file_path
        finally:
            # Clean up.
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass  # File might have been already deleted

    @staticmethod
    def _get_content(input_data: InputData) -> bytes:
        if input_data.type == InputType.URL:
            return FileService.download_from_url(input_data.data)
        if input_data.type == InputType.BASE64:
            return FileService.decode_base64(input_data.data)
        raise InvalidInputError(f"Unsupported input type: {input_data.type}")

    @staticmethod
    def process_input_data(input_data: InputData) -> Union[Image.Image, bytes]:
        """
        Process input data and return either PIL Image or bytes for video.

        Args:
            input_data: Input data specification.

        Returns:
            PIL Image for images, bytes for videos.

        Raises:
            InvalidInputError: If input is invalid.
            DownloadError: If download fails.
        """
        content = FileService._get_content(input_data)

        # Try to open as image first.
        try:
            image = Image.open(io.BytesIO(content))
            # Verify we can read basic image info (format).
            # Don't fully load the image here - that will happen during processing.
            if image.format is None:
                raise ValueError("Unknown image format")
            return image
        except Exception:
            # If not an image, assume it's a video and return as bytes.
            return content

    @staticmethod
    @contextmanager
    def process_video_input(input_data: InputData) -> Generator[str, None, None]:
        """
        Process video input and return temporary file path.

        Args:
            input_data: Input data specification.

        Yields:
            Path to temporary video file.
        """
        content = FileService._get_content(input_data)

        # Use generic suffix for video files.
        with FileService.get_temp_file(content, suffix=".mp4") as temp_path:
            yield temp_path
