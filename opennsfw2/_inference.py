"""
Inference utilities.
"""

from typing import Optional, Sequence

import numpy as np  # type: ignore
from PIL import Image  # type: ignore

from ._image import preprocess_image, Preprocessing
from ._model import get_default_weights_path, make_open_nsfw_model


def predict(
        image_paths: Sequence[str],
        batch_size: int = 32,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path()
) -> np.ndarray:
    """
    Pipeline from image paths to predictions.
    """
    images = np.array([
        preprocess_image(Image.open(image_path), preprocessing)
        for image_path in image_paths
    ])
    model = make_open_nsfw_model(weights_path=weights_path)
    return model.predict(images, batch_size=batch_size)
