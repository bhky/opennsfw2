"""
Inspection utilities.
"""

from typing import Optional

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from matplotlib import cm  # type: ignore


def make_grad_cam_heatmap(
        image: np.ndarray,
        model: tf.keras.Model,
        last_conv_layer_name: str,
        classification_linear_layer_name: str,
        prediction_index: Optional[int] = None
) -> np.ndarray:
    """
    References:
    https://keras.io/examples/vision/grad_cam/
    """
    if len(image.shape) != 3:
        raise ValueError("Input image array must have 3 dimensions.")

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output,
         model.get_layer(classification_linear_layer_name).output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, output = grad_model(np.expand_dims(image, 0))
        if prediction_index is None:
            prediction_index = tf.argmax(output[0])
        class_channel = output[:, prediction_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Shape: (num_channels,).
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Shape of last_conv_layer_output: (1, h, w, num_channels).
    # Shape of heatmap: (h, w, 1).
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalise to [0.0, 1.0].
    heatmap = tf.maximum(heatmap, 0.0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def _resize(
        image: np.array,
        target_height: int,
        target_width: int
) -> np.ndarray:
    pil_image = tf.keras.preprocessing.image.array_to_img(image)
    pil_image = pil_image.resize((target_width, target_height))
    return np.array(pil_image)


def save_grad_cam(
        image: np.ndarray,
        heatmap: np.ndarray,
        grad_cam_path: str,
        target_height: int,
        target_width: int,
        alpha: float
) -> None:
    """
    References:
    https://keras.io/examples/vision/grad_cam/
    """
    # Rescale heatmap to a range 0-255.
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap.
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap.
    # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
    jet_colors = jet(np.arange(256), bytes=True)[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Superimpose the heatmap on the input image after resizing.
    jet_heatmap = _resize(jet_heatmap, target_height, target_width)
    image = _resize(image, target_height, target_width)

    superimposed_image = jet_heatmap * alpha + image
    pil_superimposed_image = tf.keras.preprocessing.image.array_to_img(
        superimposed_image
    )

    # Save the superimposed image.
    pil_superimposed_image.save(grad_cam_path)


def make_and_save_nsfw_grad_cam(
        image: np.ndarray,
        open_nsfw_model: tf.keras.Model,
        grad_cam_path: str,
        grad_cam_height: int,
        grad_cam_width: int,
        alpha: float
) -> None:
    heatmap = make_grad_cam_heatmap(
        image, open_nsfw_model, "activation_stage3_block2", "fc_nsfw", 1
    )
    save_grad_cam(
        image, heatmap, grad_cam_path, grad_cam_height, grad_cam_width, alpha
    )
