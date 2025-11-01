"""
Custom exception classes.
"""


class OpenNSFWAPIError(Exception):
    """Base exception for OpenNSFW API errors."""


class InvalidInputError(OpenNSFWAPIError):
    """Raised when input data is invalid."""


class DownloadError(OpenNSFWAPIError):
    """Raised when file download fails."""


class ModelError(OpenNSFWAPIError):
    """Raised when model operations fail."""


class ProcessingError(OpenNSFWAPIError):
    """Raised when prediction processing fails."""
