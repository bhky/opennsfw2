"""
File service for handling input data (URLs and base64).
"""
import base64
import io
import os
import tempfile
from contextlib import contextmanager
from typing import Generator, Union
from urllib.parse import urlparse
import requests

from PIL import Image

from ..models import InputType, InputData
from ..utils.exceptions import InvalidInputError, DownloadError


class FileService:
    """Service for handling file operations and input processing."""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if a URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def is_valid_base64(data: str) -> bool:
        """Check if a string is valid base64."""
        try:
            base64.b64decode(data, validate=True)
            return True
        except Exception:
            return False
    
    @staticmethod
    def download_from_url(url: str, timeout: int = 30) -> bytes:
        """
        Download file content from URL.
        
        Args:
            url: URL to download from.
            timeout: Request timeout in seconds.
            
        Returns:
            File content as bytes.
            
        Raises:
            DownloadError: If download fails.
        """
        if not FileService.is_valid_url(url):
            raise InvalidInputError(f"Invalid URL: {url}")
        
        try:
            headers = {
                'User-Agent': 'OpenNSFW2-API/0.14.0'
            }
            response = requests.get(url, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check content length if available.
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 100 * 1024 * 1024:  # 100MB limit
                raise DownloadError("File too large (>100MB)")
            
            # Download content.
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 100 * 1024 * 1024:  # 100MB limit
                    raise DownloadError("File too large (>100MB)")
            
            return content
            
        except requests.RequestException as e:
            raise DownloadError(f"Failed to download from URL: {e}")
    
    @staticmethod
    def decode_base64(data: str) -> bytes:
        """
        Decode base64 string to bytes.
        
        Args:
            data: Base64 encoded string.
            
        Returns:
            Decoded bytes.
            
        Raises:
            InvalidInputError: If base64 is invalid.
        """
        if not FileService.is_valid_base64(data):
            raise InvalidInputError("Invalid base64 data")
        
        try:
            return base64.b64decode(data)
        except Exception as e:
            raise InvalidInputError(f"Failed to decode base64: {e}")
    
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
    def process_input_data(input_data: InputData) -> Union[Image.Image, str]:
        """
        Process input data and return either PIL Image or file path.
        
        Args:
            input_data: Input data specification.
            
        Returns:
            PIL Image for images, file path for videos.
            
        Raises:
            InvalidInputError: If input is invalid.
            DownloadError: If download fails.
        """
        if input_data.type == InputType.URL:
            content = FileService.download_from_url(input_data.data)
        elif input_data.type == InputType.BASE64:
            content = FileService.decode_base64(input_data.data)
        else:
            raise InvalidInputError(f"Unsupported input type: {input_data.type}")
        
        # Try to open as image first.
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()  # Verify it's a valid image.
            # Reopen after verify (verify() closes the file).
            image = Image.open(io.BytesIO(content))
            return image
        except Exception:
            # If not an image, assume it's a video and return as temp file.
            # We'll determine the appropriate suffix based on content.
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
        if input_data.type == InputType.URL:
            content = FileService.download_from_url(input_data.data)
        elif input_data.type == InputType.BASE64:
            content = FileService.decode_base64(input_data.data)
        else:
            raise InvalidInputError(f"Unsupported input type: {input_data.type}")
        
        # Use generic suffix for video files.
        with FileService.get_temp_file(content, suffix=".mp4") as temp_path:
            yield temp_path 