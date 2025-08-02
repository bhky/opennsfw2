"""
Validation utilities.
"""
from typing import List
from urllib.parse import urlparse

from ..models import InputData, InputType
from .exceptions import InvalidInputError


def validate_url(url: str) -> None:
    """
    Validate URL format.
    
    Args:
        url: URL string to validate.
        
    Raises:
        InvalidInputError: If URL is invalid.
    """
    if not url or not isinstance(url, str):
        raise InvalidInputError("URL must be a non-empty string")
    
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise InvalidInputError("URL must have scheme and netloc")
        
        if result.scheme not in ('http', 'https'):
            raise InvalidInputError("URL must use http or https scheme")
            
    except Exception as e:
        raise InvalidInputError(f"Invalid URL format: {e}")


def validate_base64(data: str) -> None:
    """
    Validate base64 string format.
    
    Args:
        data: Base64 string to validate.
        
    Raises:
        InvalidInputError: If base64 is invalid.
    """
    if not data or not isinstance(data, str):
        raise InvalidInputError("Base64 data must be a non-empty string")
    
    # Basic length check - base64 strings should be multiples of 4.
    if len(data) % 4 != 0:
        raise InvalidInputError("Base64 data length must be multiple of 4")
    
    # Check for invalid characters.
    import base64
    try:
        base64.b64decode(data, validate=True)
    except Exception as e:
        raise InvalidInputError(f"Invalid base64 format: {e}")


def validate_input_data(input_data: InputData) -> None:
    """
    Validate input data specification.
    
    Args:
        input_data: Input data to validate.
        
    Raises:
        InvalidInputError: If input data is invalid.
    """
    if input_data.type == InputType.URL:
        validate_url(input_data.data)
    elif input_data.type == InputType.BASE64:
        validate_base64(input_data.data)
    else:
        raise InvalidInputError(f"Unsupported input type: {input_data.type}")


def validate_input_list(inputs: List[InputData]) -> None:
    """
    Validate list of input data.
    
    Args:
        inputs: List of input data to validate.
        
    Raises:
        InvalidInputError: If any input data is invalid.
    """
    if not inputs:
        raise InvalidInputError("Input list cannot be empty")
    
    if len(inputs) > 50:  # Reasonable limit.
        raise InvalidInputError("Too many inputs (maximum 50)")
    
    for i, input_data in enumerate(inputs):
        try:
            validate_input_data(input_data)
        except InvalidInputError as e:
            raise InvalidInputError(f"Input {i+1}: {e}") 