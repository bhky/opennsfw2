"""
Prediction router for NSFW detection endpoints.
"""
import time
from typing import List

from fastapi import APIRouter, HTTPException, status
from PIL import Image

from ..models import (
    SingleImageRequest, SingleImageResponse, 
    MultipleImagesRequest, MultipleImagesResponse,
    VideoRequest, VideoResponse,
    ErrorResponse, PredictionResult, VideoResult
)
from ..services.prediction_service import PredictionService
from ..services.file_service import FileService
from ..utils.exceptions import OpenNSFWAPIError, InvalidInputError, DownloadError, ProcessingError

router = APIRouter()


@router.post(
    "/image", 
    response_model=SingleImageResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_image(request: SingleImageRequest) -> SingleImageResponse:
    """Predict NSFW probability for a single image."""
    start_time = time.time()
    
    try:
        # Get prediction service.
        service = PredictionService()
        
        # Process input data.
        processed_input = FileService.process_input_data(request.input)
        
        # Ensure it's an image.
        if not isinstance(processed_input, Image.Image):
            raise InvalidInputError("Input is not a valid image")
        
        # Get prediction.
        sfw_prob, nsfw_prob = service.predict_image(
            processed_input, 
            preprocessing=request.options.preprocessing if request.options else None
        )
        
        # Calculate processing time.
        processing_time = (time.time() - start_time) * 1000
        
        return SingleImageResponse(
            result=PredictionResult(
                nsfw_probability=nsfw_prob,
                sfw_probability=sfw_prob
            ),
            processing_time_ms=processing_time,
            model_version="0.14.0"
        )
        
    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Download failed: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        )


@router.post(
    "/images", 
    response_model=MultipleImagesResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_images(request: MultipleImagesRequest) -> MultipleImagesResponse:
    """Predict NSFW probabilities for multiple images."""
    start_time = time.time()
    
    try:
        # Get prediction service.
        service = PredictionService()
        
        # Process all input data.
        processed_inputs: List[Image.Image] = []
        for input_data in request.inputs:
            processed_input = FileService.process_input_data(input_data)
            
            # Ensure it's an image.
            if not isinstance(processed_input, Image.Image):
                raise InvalidInputError(f"Input {input_data.data[:50]}... is not a valid image")
            
            processed_inputs.append(processed_input)
        
        # Get predictions.
        predictions = service.predict_images(
            processed_inputs,
            preprocessing=request.options.preprocessing if request.options else None
        )
        
        # Convert to results.
        results = []
        for sfw_prob, nsfw_prob in predictions:
            results.append(PredictionResult(
                nsfw_probability=nsfw_prob,
                sfw_probability=sfw_prob
            ))
        
        # Calculate processing time.
        processing_time = (time.time() - start_time) * 1000
        
        return MultipleImagesResponse(
            results=results,
            processing_time_ms=processing_time,
            model_version="0.14.0"
        )
        
    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Download failed: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        )


@router.post(
    "/video", 
    response_model=VideoResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict_video(request: VideoRequest) -> VideoResponse:
    """Predict NSFW probabilities for video frames."""
    start_time = time.time()
    
    try:
        # Get prediction service.
        service = PredictionService()
        
        # Process video input and get temporary file.
        with FileService.process_video_input(request.input) as video_path:
            # Get predictions.
            elapsed_seconds, nsfw_probabilities = service.predict_video(
                video_path,
                preprocessing=request.options.preprocessing if request.options else None,
                frame_interval=request.options.frame_interval if request.options else 8,
                aggregation_size=request.options.aggregation_size if request.options else 8,
                aggregation=request.options.aggregation if request.options else None
            )
        
        # Calculate processing time.
        processing_time = (time.time() - start_time) * 1000
        
        return VideoResponse(
            result=VideoResult(
                elapsed_seconds=elapsed_seconds,
                nsfw_probabilities=nsfw_probabilities
            ),
            processing_time_ms=processing_time,
            model_version="0.14.0"
        )
        
    except InvalidInputError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DownloadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Download failed: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}"
        ) 