"""
Image utilities.
"""
import io
from enum import Enum

import keras  # type: ignore
import numpy as np
import skimage.io  # type: ignore
from PIL import Image  # type: ignore

from ._typing import KerasTensor, NDFloat32Array


class Preprocessing(str, Enum):
    YAHOO = "YAHOO"
    SIMPLE = "SIMPLE"


def preprocess_image(
        pil_image: Image.Image,
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
        pil_image_resized = pil_image.resize(
            (256, 256), resample=Image.BILINEAR  # pylint: disable=no-member
        )

        fh_im = io.BytesIO()
        pil_image_resized.save(fh_im, format="JPEG")
        fh_im.seek(0)

        image: NDFloat32Array = skimage.io.imread(  # type: ignore
            fh_im, as_gray=False
        ).astype(np.float32)

        height, width, _ = image.shape
        h, w = (224, 224)

        h_off = max((height - h) // 2, 0)
        w_off = max((width - w) // 2, 0)
        image = image[h_off:h_off + h, w_off:w_off + w, :]

    elif preprocessing == Preprocessing.SIMPLE:
        pil_image_resized = pil_image.resize(
            (224, 224), resample=Image.BILINEAR  # pylint: disable=no-member
        )
        image = np.array(pil_image_resized).astype(np.float32)

    # RGB to BGR
    image = image[:, :, ::-1]

    # Subtract the training dataset mean value of each channel.
    vgg_mean = [104, 117, 123]
    image = image - np.array(vgg_mean, dtype=np.float32)

    return image


def preprocess_image_tensor(
        image: KerasTensor,
        preprocessing: Preprocessing = Preprocessing.YAHOO
) -> KerasTensor:
    """
    Tensor-based preprocessing equivalent of `preprocess_image`, suitable
    for use with dataset pipelines (e.g., tf.data.Dataset.map).

    Expects a single uint8 tensor of shape (H, W, C) in RGB channel order.
    The JPEG round-trip from the YAHOO pipeline is intentionally omitted.
    """
    image = keras.ops.cast(image, "float32")

    if preprocessing == Preprocessing.YAHOO:
        image = keras.ops.image.resize(image, (256, 256), interpolation="bilinear")
        h_off = (256 - 224) // 2
        w_off = (256 - 224) // 2
        image = image[h_off:h_off + 224, w_off:w_off + 224, :]  # type: ignore

    elif preprocessing == Preprocessing.SIMPLE:
        image = keras.ops.image.resize(image, (224, 224), interpolation="bilinear")

    # RGB to BGR.
    image = image[..., ::-1]  # type: ignore

    vgg_mean = keras.ops.cast([104, 117, 123], "float32")
    return image - vgg_mean  # type: ignore
