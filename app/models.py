"""
Pydantic models for request and response validation.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class InputType(str, Enum):
    """Supported input types."""
    URL = "url"
    BASE64 = "base64"


class PreprocessingType(str, Enum):
    """Preprocessing options."""
    YAHOO = "YAHOO"
    SIMPLE = "SIMPLE"


class AggregationType(str, Enum):
    """Video aggregation methods."""
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    MAX = "MAX"
    MIN = "MIN"


class InputData(BaseModel):
    """Input data specification."""
    type: InputType = Field(..., description="Type of input data")
    data: str = Field(..., description="URL or base64 encoded data")


class ImageOptions(BaseModel):
    """Options for image prediction."""
    preprocessing: PreprocessingType = Field(
        default=PreprocessingType.YAHOO,
        description="Preprocessing method to use"
    )


class VideoOptions(BaseModel):
    """Options for video prediction."""
    preprocessing: PreprocessingType = Field(
        default=PreprocessingType.YAHOO,
        description="Preprocessing method to use"
    )
    frame_interval: int = Field(
        default=8,
        ge=1,
        description="Prediction will be done on every this number of frames"
    )
    aggregation_size: int = Field(
        default=8,
        ge=1,
        description="Number of frames for aggregation"
    )
    aggregation: AggregationType = Field(
        default=AggregationType.MEAN,
        description="Aggregation method"
    )


class SingleImageRequest(BaseModel):
    """Request model for single image prediction."""
    input: InputData = Field(..., description="Input image data")
    options: Optional[ImageOptions] = Field(
        default_factory=ImageOptions,
        description="Prediction options"
    )


class MultipleImagesRequest(BaseModel):
    """Request model for multiple images prediction."""
    inputs: List[InputData] = Field(..., description="List of input image data")
    options: Optional[ImageOptions] = Field(
        default_factory=ImageOptions,
        description="Prediction options"
    )

    @validator('inputs')
    @classmethod
    def inputs_not_empty(cls, v: List[InputData]) -> List[InputData]:
        """Validate that inputs list is not empty."""
        if not v:
            raise ValueError('inputs cannot be empty')
        return v


class VideoRequest(BaseModel):
    """Request model for video prediction."""
    input: InputData = Field(..., description="Input video data")
    options: Optional[VideoOptions] = Field(
        default_factory=VideoOptions,
        description="Prediction options"
    )


class PredictionResult(BaseModel):
    """Single prediction result."""
    nsfw_probability: float = Field(..., description="NSFW probability")
    sfw_probability: float = Field(..., description="SFW probability")


class SingleImageResponse(BaseModel):
    """Response model for single image prediction."""
    success: bool = Field(True, description="Success status")
    result: PredictionResult = Field(..., description="Prediction result")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version")


class MultipleImagesResponse(BaseModel):
    """Response model for multiple images prediction."""
    success: bool = Field(True, description="Success status")
    results: List[PredictionResult] = Field(..., description="Prediction results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version")


class VideoResult(BaseModel):
    """Video prediction result."""
    elapsed_seconds: List[float] = Field(..., description="Elapsed time for each frame")
    nsfw_probabilities: List[float] = Field(..., description="NSFW probability for each frame")


class VideoResponse(BaseModel):
    """Response model for video prediction."""
    success: bool = Field(True, description="Success status")
    result: VideoResult = Field(..., description="Video prediction result")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Success status")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
