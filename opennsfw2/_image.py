"""
Image utilities.
"""
import io
from enum import auto, Enum

import numpy as np
import skimage.io  # type: ignore
from PIL import Image  # type: ignore

from ._typing import NDFloat32Array


class Preprocessing(Enum):
    YAHOO = auto()
    SIMPLE = auto()


def preprocess_image(
        pil_image: Image,
        preprocessing: Preprocessing = Preprocessing.YAHOO
) -> NDFloat32Array:
    """
    Preprocessing for the pre-trained Open NSFW model weights.

    References:
    https://github.com/mdietrichstein/tensorflow-open_nsfw
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    if preprocessing == Preprocessing.YAHOO:
        pil_image_resized = pil_image.resize((256, 256), resample=Image.BILINEAR)

        fh_im = io.BytesIO()
        pil_image_resized.save(fh_im, format="JPEG")
        fh_im.seek(0)

        image: NDFloat32Array = skimage.io.imread(
            fh_im, as_gray=False
        ).astype(np.float32)

        height, width, _ = image.shape
        h, w = (224, 224)

        h_off = max((height - h) // 2, 0)
        w_off = max((width - w) // 2, 0)
        image = image[h_off:h_off + h, w_off:w_off + w, :]

    elif preprocessing == Preprocessing.SIMPLE:
        pil_image_resized = pil_image.resize((224, 224), resample=Image.BILINEAR)
        image = np.array(pil_image_resized).astype(np.float32)

    else:
        raise ValueError("Invalid preprocessing option.")

    # RGB to BGR
    image = image[:, :, ::-1]

    # Subtract the training dataset mean value of each channel.
    vgg_mean = [104, 117, 123]
    image = image - np.array(vgg_mean, dtype=np.float32)

    return image
