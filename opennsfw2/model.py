"""
References:
https://github.com/mdietrichstein/tensorflow-open_nsfw
https://github.com/yahoo/open_nsfw
"""

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

WEIGHTS: Dict[str, Dict[str, np.ndarray]] = np.load(
    "../weights/open_nsfw-weights.npy", encoding="latin1", allow_pickle=True
).item()


def _get_weights(layer_name: str, field_name: str) -> np.ndarray:
    if layer_name not in WEIGHTS:
        raise ValueError(f"No weights found for layer {layer_name}.")

    w = WEIGHTS[layer_name]
    if field_name not in w:
        raise ValueError(f"No field {field_name} in layer {layer_name}.")

    return w[field_name]


def _fully_connected(name: str, units: int) -> layers.Dense:
    return layers.Dense(
        name=name,
        units=units,
        kernel_initializer=tf.constant_initializer(
            _get_weights(name, "weights"), dtype=tf.float32
        ),
        bias_initializer=tf.constant_initializer(
            _get_weights(name, "biases"), dtype=tf.float32
        )
    )


def _conv2d(
        name: str,
        num_filters: int,
        kernel_size: int,
        stride: int,
        padding: str = "same"
) -> layers.Conv2D:
    return layers.Conv2D(
        name=name,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        kernel_initializer=tf.constant_initializer(
            _get_weights(name, "weights"), dtype=tf.float32
        ),
        bias_initializer=tf.constant_initializer(
            _get_weights(name, "biases"), dtype=tf.float32
        )
    )


def _batch_norm(name: str) -> layers.BatchNormalization:
    return layers.BatchNormalization(
        name=name,
        epsilon=1e-05,  # Default used in Caffe.
        gamma_initializer=tf.constant_initializer(
            _get_weights(name, "scale"), dtype=tf.float32
        ),
        beta_initializer=tf.constant_initializer(
            _get_weights(name, "offset"), dtype=tf.float32
        ),
        moving_mean_initializer=tf.constant_initializer(
            _get_weights(name, "mean"), dtype=tf.float32
        ),
        moving_variance_initializer=tf.constant_initializer(
            _get_weights(name, "variance"), dtype=tf.float32
        ),
    )


def _conv_block(
        inputs: tf.Tensor,
        stage: int,
        block: int,
        nums_filters: Tuple[int, int, int],
        kernel_size: int = 3,
        stride: int = 2,
) -> tf.Tensor:
    num_filters_1, num_filters_2, num_filters_3 = nums_filters

    conv_name_base = f"conv_stage{stage}_block{block}_branch"
    bn_name_base = f"bn_stage{stage}_block{block}_branch"
    shortcut_name_post = f"_stage{stage}_block{block}_proj_shortcut"

    shortcut = _conv2d(
        name=f"conv{shortcut_name_post}",
        num_filters=num_filters_3,
        kernel_size=1,
        stride=stride,
        padding="same"
    )(inputs)

    shortcut = _batch_norm(f"bn{shortcut_name_post}")(shortcut)

    x = _conv2d(
        name=f"{conv_name_base}2a",
        num_filters=num_filters_1,
        kernel_size=1,
        stride=stride,
        padding="same"
    )(inputs)
    x = _batch_norm(f"{bn_name_base}2a")(x)
    x = tf.nn.relu(x)

    x = _conv2d(
        name=f"{conv_name_base}2b",
        num_filters=num_filters_2,
        kernel_size=kernel_size,
        stride=1,
        padding="same"
    )(x)
    x = _batch_norm(f"{bn_name_base}2b")(x)
    x = tf.nn.relu(x)

    x = _conv2d(
        name=f"{conv_name_base}2c",
        num_filters=num_filters_3,
        kernel_size=1,
        stride=1,
        padding="same"
    )(x)
    x = _batch_norm(f"{bn_name_base}2c")(x)

    x = layers.Add()([x, shortcut])

    return tf.nn.relu(x)


def _identity_block(
        inputs: tf.Tensor,
        stage: int,
        block: int,
        nums_filters: Tuple[int, int, int],
        kernel_size: int
) -> tf.Tensor:
    num_filters_1, num_filters_2, num_filters_3 = nums_filters

    conv_name_base = f"conv_stage{stage}_block{block}_branch"
    bn_name_base = f"bn_stage{stage}_block{block}_branch"

    x = _conv2d(
        name=f"{conv_name_base}2a",
        num_filters=num_filters_1,
        kernel_size=1,
        stride=1,
        padding="same"
    )(inputs)
    x = _batch_norm(f"{bn_name_base}2a")(x)
    x = tf.nn.relu(x)

    x = _conv2d(
        name=f"{conv_name_base}2b",
        num_filters=num_filters_2,
        kernel_size=kernel_size,
        stride=1,
        padding="same"
    )(x)
    x = _batch_norm(f"{bn_name_base}2b")(x)
    x = tf.nn.relu(x)

    x = _conv2d(
        name=f"{conv_name_base}2c",
        num_filters=num_filters_3,
        kernel_size=1,
        stride=1,
        padding="same"
    )(x)
    x = _batch_norm(f"{bn_name_base}2c")(x)

    x = layers.Add()([x, inputs])

    return tf.nn.relu(x)
