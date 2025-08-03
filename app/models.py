"""
Pydantic models for request and response validation.
"""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class InputType(str, Enum):
    URL = "url"
    BASE64 = "base64"


class PreprocessingType(str, Enum):
    YAHOO = "YAHOO"
    SIMPLE = "SIMPLE"


class AggregationType(str, Enum):
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    MAX = "MAX"
    MIN = "MIN"


class InputData(BaseModel):
    type: InputType = Field(..., description="Type of input data")
    data: str = Field(..., description="URL or base64 data")


class ImageOptions(BaseModel):
    """Options for image prediction."""
    preprocessing: PreprocessingType = Field(
        default=PreprocessingType.YAHOO,
        description="Preprocessing method"
    )


class VideoOptions(BaseModel):
    """Options for video prediction."""
    preprocessing: PreprocessingType = Field(
        default=PreprocessingType.YAHOO,
        description="Preprocessing method"
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
    nsfw_probability: float = Field(..., description="NSFW probability")
    sfw_probability: float = Field(..., description="SFW probability")


class VideoResult(BaseModel):
    """Result model for video prediction."""
    elapsed_seconds: List[float] = Field(..., description="Elapsed seconds for each frame")
    nsfw_probabilities: List[float] = Field(..., description="NSFW probabilities for each frame")


class SingleImageResponse(BaseModel):
    """Response model for single image prediction."""
    success: bool = Field(..., description="Whether the prediction was successful")
    result: PredictionResult = Field(..., description="Prediction result")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    version: str = Field(..., description="OpenNSFW2 package version")


class MultipleImagesResponse(BaseModel):
    """Response model for multiple images prediction."""
    success: bool = Field(..., description="Whether the prediction was successful")
    results: List[PredictionResult] = Field(..., description="Prediction results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    version: str = Field(..., description="OpenNSFW2 package version")


class VideoResponse(BaseModel):
    """Response model for video prediction."""
    success: bool = Field(..., description="Whether the prediction was successful")
    result: VideoResult = Field(..., description="Prediction result")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    version: str = Field(..., description="OpenNSFW2 package version")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")
