"""
Inspection utilities.
"""
from typing import Optional

import numpy as np
import tensorflow as tf  # type: ignore
from matplotlib import cm  # type: ignore
from PIL import Image  # type: ignore

from ._image import preprocess_image, Preprocessing
from ._typing import NDUInt8Array, NDFloat32Array


def make_grad_cam_heatmap(
        preprocessed_image: NDFloat32Array,
        model: tf.keras.Model,
        last_conv_layer_name: str,
        classification_linear_layer_name: str,
        prediction_index: Optional[int] = None
) -> NDFloat32Array:
    """
    References:
    https://keras.io/examples/vision/grad_cam/
    """
    if len(preprocessed_image.shape) != 3:
        raise ValueError(
            "Input preprocessed image array must have 3 dimensions."
        )

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output,
         model.get_layer(classification_linear_layer_name).output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, output = grad_model(
            np.expand_dims(preprocessed_image, 0)
        )
        if prediction_index is None:
            prediction_index = tf.argmax(output[0])
        class_channel = output[:, prediction_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Shape: (num_channels,).
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Shape of last_conv_layer_output: (1, h, w, num_channels).
    # Shape of heatmap: (h, w, 1).
    tf_heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    tf_heatmap = tf.squeeze(tf_heatmap)

    # Normalise to [0.0, 1.0].
    tf_heatmap = tf.maximum(tf_heatmap, 0.0) / tf.reduce_max(tf_heatmap)
    heatmap: NDFloat32Array = tf_heatmap.numpy()
    return heatmap


def _resize(
        image: NDUInt8Array,
        target_height: int,
        target_width: int
) -> NDUInt8Array:
    pil_image = tf.keras.preprocessing.image.array_to_img(image)
    pil_image = pil_image.resize((target_width, target_height))
    return np.array(pil_image)


def save_grad_cam(
        pil_image: Image,
        heatmap: NDFloat32Array,
        grad_cam_path: str,
        alpha: float
) -> None:
    """
    References:
    https://keras.io/examples/vision/grad_cam/
    """
    # Rescale heatmap to a range 0-255.
    scaled_heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap.
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap.
    # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
    jet_colors = jet(np.arange(256), bytes=True)[:, :3]
    jet_heatmap = jet_colors[scaled_heatmap]

    # Superimpose the heatmap on the input image after resizing.
    jet_heatmap = _resize(jet_heatmap, pil_image.height, pil_image.width)

    superimposed_image = jet_heatmap * alpha + np.array(pil_image)
    pil_superimposed_image = tf.keras.preprocessing.image.array_to_img(
        superimposed_image
    )

    # Save the superimposed image.
    pil_superimposed_image.save(grad_cam_path)


def make_and_save_nsfw_grad_cam(
        pil_image: Image,
        preprocessing: Preprocessing,
        open_nsfw_model: tf.keras.Model,
        grad_cam_path: str,
        alpha: float
) -> None:
    heatmap = make_grad_cam_heatmap(
        preprocess_image(pil_image, preprocessing), open_nsfw_model,
        "activation_stage3_block2", "fc_nsfw", 1
    )
    save_grad_cam(pil_image, heatmap, grad_cam_path, alpha)
