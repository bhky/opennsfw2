"""
Prediction service for OpenNSFW2 model.
"""
from typing import List, Optional, Tuple, Union, Sequence
from threading import Lock

import opennsfw2 as n2
from keras import Model
from PIL import Image

from ..pydantic_models import Aggregation, Preprocessing


class PredictionService:
    """Singleton service for managing the NSFW model and predictions."""

    _instance: Optional["PredictionService"] = None
    _lock = Lock()
    _model: Optional[Model] = None
    _model_loaded: bool = False

    def __new__(cls) -> "PredictionService":
        # Double-Checked locking.
        if cls._instance is None:
            with cls._lock:
                # Double-check for thread safety.
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self) -> None:
        """Initialize the model."""
        try:
            self._model = n2.make_open_nsfw_model()
            self._model_loaded = True
        except Exception as e:
            self._model_loaded = False
            raise RuntimeError(f"Failed to load NSFW model: {e}") from e

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded

    def predict_image(
        self,
        image: Union[str, Image.Image],
        preprocessing: Preprocessing = Preprocessing.YAHOO
    ) -> float:
        """
        Predict NSFW probability for a single image.

        Args:
            image: PIL Image object or path to image file.
            preprocessing: Preprocessing method to use.

        Returns:
            NSFW probability.
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded")

        return n2.predict_image(image, preprocessing=preprocessing)

    def predict_images(
        self,
        images: Sequence[Union[str, Image.Image]],
        preprocessing: Preprocessing = Preprocessing.YAHOO
    ) -> List[float]:
        """
        Predict NSFW probabilities for multiple images.

        Args:
            images: List of PIL Image objects or paths to image files.
            preprocessing: Preprocessing method to use.

        Returns:
            List of NSFW probabilities.
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded")

        return n2.predict_images(images, preprocessing=preprocessing)

    def predict_video(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        video_path: str,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        frame_interval: int = 8,
        aggregation_size: int = 8,
        aggregation: Aggregation = Aggregation.MEAN,
        nsfw_threshold: float = 0.80,
        early_termination_count: int = 10
    ) -> Tuple[List[float], List[float], dict]:
        """
        Predict NSFW probabilities for video frames with early termination.

        Args:
            video_path: Path to video file.
            preprocessing: Preprocessing method to use.
            frame_interval: Prediction interval.
            aggregation_size: Number of frames for aggregation.
            aggregation: Aggregation method.
            nsfw_threshold: NSFW probability threshold for flagging (default 0.80).
            early_termination_count: Stop processing after this many flagged frames (default 10).

        Returns:
            Tuple of (elapsed_seconds, nsfw_probabilities, usage_info).
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded")

        # Get predictions with early termination.
        elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(
            video_path,
            frame_interval=frame_interval,
            aggregation_size=aggregation_size,
            aggregation=aggregation,
            preprocessing=preprocessing,
            progress_bar=False  # Disable progress bar for API.
        )

        # Early termination logic: stop if too many frames are flagged.
        flagged_count = 0
        early_terminated = False
        processed_frames = len(nsfw_probabilities)

        for i, prob in enumerate(nsfw_probabilities):
            if prob >= nsfw_threshold:
                flagged_count += 1
                if flagged_count >= early_termination_count:
                    # Terminate early - trim results to current position.
                    elapsed_seconds = elapsed_seconds[:i+1]
                    nsfw_probabilities = nsfw_probabilities[:i+1]
                    processed_frames = i + 1
                    early_terminated = True
                    break

        # Build usage info.
        usage_info = {
            "frames_processed": processed_frames,
            "frames_flagged": flagged_count,
            "early_terminated": early_terminated,
            "nsfw_threshold": nsfw_threshold,
            "early_termination_trigger": early_termination_count if early_terminated else None
        }

        return elapsed_seconds, nsfw_probabilities, usage_info
