"""
Image utilities.
"""

import io
from enum import auto, Enum

import numpy as np  # type: ignore
import skimage.io  # type: ignore
from PIL import Image  # type: ignore


class Preprocessing(Enum):
    SIMPLE = auto()
    YAHOO = auto()


def load_and_preprocess_image(
        image_path: str,
        preprocessing: Preprocessing = Preprocessing.YAHOO
) -> np.ndarray:
    """
    Load image from image_path with preprocessing for the
    pre-trained Open NSFW model weights.

    References:
    https://github.com/mdietrichstein/tensorflow-open_nsfw
    """
    pil_image = Image.open(image_path)

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    if preprocessing == Preprocessing.YAHOO:
        pil_image_resized = pil_image.resize((256, 256), resample=Image.BILINEAR)

        fh_im = io.BytesIO()
        pil_image_resized.save(fh_im, format="JPEG")
        fh_im.seek(0)

        image = skimage.io.imread(fh_im, as_gray=False).astype(np.float32)

        H, W, _ = image.shape
        h, w = (224, 224)

        h_off = max((H - h) // 2, 0)
        w_off = max((W - w) // 2, 0)
        image = image[h_off:h_off + h, w_off:w_off + w, :]

    elif preprocessing == Preprocessing.SIMPLE:
        pil_image_resized = pil_image.resize((224, 224), resample=Image.BILINEAR)
        image = np.array(pil_image_resized).astype(np.float32)

    else:
        raise ValueError("Invalid preprocessing option.")

    # RGB to BGR
    image = image[:, :, ::-1]

    # Subtract the training dataset mean value in each channel.
    vgg_mean = [104, 117, 123]
    image = image - np.array(vgg_mean, dtype=np.float32)

    return image
