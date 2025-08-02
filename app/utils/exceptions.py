"""
Custom exception classes.
"""


class OpenNSFWAPIError(Exception):
    """Base exception for OpenNSFW API errors."""
    pass


class InvalidInputError(OpenNSFWAPIError):
    """Raised when input data is invalid."""
    pass


class DownloadError(OpenNSFWAPIError):
    """Raised when file download fails."""
    pass


class ModelError(OpenNSFWAPIError):
    """Raised when model operations fail."""
    pass


class ProcessingError(OpenNSFWAPIError):
    """Raised when prediction processing fails."""
    pass 