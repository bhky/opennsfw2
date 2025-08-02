"""
Prediction service for managing model and making predictions.
"""
from typing import List, Optional, Tuple, Union, Sequence
from threading import Lock

from PIL import Image
import opennsfw2 as n2

from ..models import PreprocessingType, AggregationType


class PredictionService:
    """Singleton service for managing the NSFW model and predictions."""

    _instance: Optional['PredictionService'] = None
    _lock = Lock()
    _model: Optional[object] = None
    _model_loaded: bool = False

    def __new__(cls) -> 'PredictionService':
        if cls._instance is None:
            with cls._lock:
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
        preprocessing: PreprocessingType = PreprocessingType.YAHOO
    ) -> Tuple[float, float]:
        """
        Predict NSFW probability for a single image.

        Args:
            image: PIL Image object or path to image file.
            preprocessing: Preprocessing method to use.

        Returns:
            Tuple of (sfw_probability, nsfw_probability).
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded")

        # Convert preprocessing enum to opennsfw2 enum.
        preprocessing_method = (
            n2.Preprocessing.YAHOO if preprocessing == PreprocessingType.YAHOO
            else n2.Preprocessing.SIMPLE
        )

        # Get prediction.
        nsfw_prob = n2.predict_image(
            image,
            preprocessing=preprocessing_method
        )
        sfw_prob = 1.0 - nsfw_prob

        return sfw_prob, nsfw_prob

    def predict_images(
        self,
        images: Sequence[Union[str, Image.Image]],
        preprocessing: PreprocessingType = PreprocessingType.YAHOO
    ) -> List[Tuple[float, float]]:
        """
        Predict NSFW probabilities for multiple images.

        Args:
            images: List of PIL Image objects or paths to image files.
            preprocessing: Preprocessing method to use.

        Returns:
            List of tuples of (sfw_probability, nsfw_probability).
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded")

        # Convert preprocessing enum to opennsfw2 enum.
        preprocessing_method = (
            n2.Preprocessing.YAHOO if preprocessing == PreprocessingType.YAHOO
            else n2.Preprocessing.SIMPLE
        )

        # Get predictions.
        nsfw_probs = n2.predict_images(
            images,  # type: ignore[arg-type]
            preprocessing=preprocessing_method
        )

        # Convert to (sfw, nsfw) tuples.
        results = []
        for nsfw_prob in nsfw_probs:
            sfw_prob = 1.0 - nsfw_prob
            results.append((sfw_prob, nsfw_prob))

        return results

    def predict_video(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        video_path: str,
        preprocessing: PreprocessingType = PreprocessingType.YAHOO,
        frame_interval: int = 8,
        aggregation_size: int = 8,
        aggregation: AggregationType = AggregationType.MEAN
    ) -> Tuple[List[float], List[float]]:
        """
        Predict NSFW probabilities for video frames.

        Args:
            video_path: Path to video file.
            preprocessing: Preprocessing method to use.
            frame_interval: Prediction interval.
            aggregation_size: Number of frames for aggregation.
            aggregation: Aggregation method.

        Returns:
            Tuple of (elapsed_seconds, nsfw_probabilities).
        """
        if not self._model_loaded:
            raise RuntimeError("Model is not loaded")

        # Convert enums to opennsfw2 enums.
        preprocessing_method = (
            n2.Preprocessing.YAHOO if preprocessing == PreprocessingType.YAHOO
            else n2.Preprocessing.SIMPLE
        )

        aggregation_method = getattr(n2.Aggregation, aggregation.value)

        # Get predictions.
        elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(
            video_path,
            frame_interval=frame_interval,
            aggregation_size=aggregation_size,
            aggregation=aggregation_method,
            preprocessing=preprocessing_method,
            progress_bar=False  # Disable progress bar for API
        )

        return elapsed_seconds, nsfw_probabilities
