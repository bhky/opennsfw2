"""
Image utilities.
"""

import io

import numpy as np
import skimage.io
from PIL import Image


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load image from url with special preprocessing for the
    pre-trained Open NSFW model weights.

    References:
    https://github.com/mdietrichstein/tensorflow-open_nsfw
    """
    pil_image = Image.open(image_path)

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

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

    # RGB to BGR
    image = image[:, :, :: -1]

    vgg_mean = [104, 117, 123]
    image = image - np.array(vgg_mean, dtype=np.float32)

    return image
