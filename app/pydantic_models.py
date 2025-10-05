"""
Pydantic models for request and response validation.
"""
from enum import Enum
from typing import List, Optional

from opennsfw2 import Aggregation, Preprocessing
from pydantic import BaseModel, Field, field_validator


class InputType(str, Enum):
    URL = "url"
    BASE64 = "base64"


class InputData(BaseModel):
    type: InputType = Field(..., description="Type of input data")
    data: str = Field(..., description="URL or base64 data")


class ImageOptions(BaseModel):
    preprocessing: Preprocessing = Field(
        default=Preprocessing.YAHOO,
        description="Preprocessing method"
    )


class VideoOptions(BaseModel):
    preprocessing: Preprocessing = Field(
        default=Preprocessing.YAHOO,
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
    aggregation: Aggregation = Field(
        default=Aggregation.MEAN,
        description="Aggregation method"
    )


class SingleImageRequest(BaseModel):
    input: InputData = Field(..., description="Input image data")
    options: Optional[ImageOptions] = Field(
        default_factory=ImageOptions,
        description="Prediction options"
    )


class MultipleImagesRequest(BaseModel):
    inputs: List[InputData] = Field(..., description="List of input image data")
    options: Optional[ImageOptions] = Field(
        default_factory=ImageOptions,
        description="Prediction options"
    )

    @field_validator("inputs", mode="before")
    @classmethod
    def inputs_not_empty(cls, v: List[InputData]) -> List[InputData]:
        """Validate that inputs list is not empty."""
        if not v:
            raise ValueError("inputs cannot be empty")
        return v


class VideoRequest(BaseModel):
    input: InputData = Field(..., description="Input video data")
    options: Optional[VideoOptions] = Field(
        default_factory=VideoOptions,
        description="Prediction options"
    )


class PredictionResult(BaseModel):
    nsfw_probability: float = Field(..., description="NSFW probability")
    sfw_probability: float = Field(..., description="SFW probability")


class VideoResult(BaseModel):
    elapsed_seconds: List[float] = Field(..., description="Elapsed seconds for each frame")
    nsfw_probabilities: List[float] = Field(..., description="NSFW probabilities for each frame")


class SingleImageResponse(BaseModel):
    success: bool = Field(..., description="Whether the prediction was successful")
    result: PredictionResult = Field(..., description="Prediction result")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    version: str = Field(..., description="OpenNSFW2 package version")


class MultipleImagesResponse(BaseModel):
    success: bool = Field(..., description="Whether the prediction was successful")
    results: List[PredictionResult] = Field(..., description="Prediction results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    version: str = Field(..., description="OpenNSFW2 package version")


class VideoResponse(BaseModel):
    success: bool = Field(..., description="Whether the prediction was successful")
    result: VideoResult = Field(..., description="Prediction result")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    version: str = Field(..., description="OpenNSFW2 package version")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
