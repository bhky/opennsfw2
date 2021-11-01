"""
References:
https://github.com/mdietrichstein/tensorflow-open_nsfw
https://github.com/yahoo/open_nsfw
"""

from abc import ABC
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class OpenNSFWModel(tf.keras.Model, ABC):

    def __init__(self) -> None:
        super().__init__()
        self._bn_epsilon = 1e-05  # Default used in Caffe.
        self._weights: Dict[str, Dict[str, np.ndarray]] = {}

    def load_weights_from_npy(self, path: str) -> None:
        self._weights = np.load(
            path, encoding="latin1", allow_pickle=True
        ).item()

    def _get_weights(self, layer_name: str, field_name: str) -> np.ndarray:
        if layer_name not in self._weights:
            raise ValueError(f"No weights found for layer {layer_name}.")

        w = self._weights[layer_name]
        if field_name not in w:
            raise ValueError(f"No field {field_name} in layer {layer_name}.")

        return w[field_name]

    def _fully_connected(self, name: str, units: int) -> layers.Dense:
        return layers.Dense(
            name=name,
            units=units,
            kernel_initializer=tf.constant_initializer(
                self._get_weights(name, "weights"), dtype=tf.float32
            ),
            bias_initializer=tf.constant_initializer(
                self._get_weights(name, "biases"), dtype=tf.float32
            )
        )

    def _conv2d(
            self,
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
                self._get_weights(name, "weights"), dtype=tf.float32
            ),
            bias_initializer=tf.constant_initializer(
                self._get_weights(name, "biases"), dtype=tf.float32
            )
        )

    def _batch_norm(self, name: str) -> layers.BatchNormalization:
        return layers.BatchNormalization(
            name=name,
            epsilon=self._bn_epsilon,
            gamma_initializer=tf.constant_initializer(
                self._get_weights(name, "scale"), dtype=tf.float32
            ),
            beta_initializer=tf.constant_initializer(
                self._get_weights(name, "offset"), dtype=tf.float32
            ),
            moving_mean_initializer=tf.constant_initializer(
                self._get_weights(name, "mean"), dtype=tf.float32
            ),
            moving_variance_initializer=tf.constant_initializer(
                self._get_weights(name, "variance"), dtype=tf.float32
            ),
        )

    def _conv_block(
            self,
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

        shortcut = self._conv2d(
            name=f"conv{shortcut_name_post}",
            num_filters=num_filters_3,
            kernel_size=1,
            stride=stride,
            padding="same"
        )(inputs)

        shortcut = self._batch_norm(f"bn{shortcut_name_post}")(shortcut)

        x = self._conv2d(
            name=f"{conv_name_base}2a",
            num_filters=num_filters_1,
            kernel_size=1,
            stride=stride,
            padding="same"
        )(inputs)
        x = self._batch_norm(f"{bn_name_base}2a")(x)
        x = tf.nn.relu(x)

        x = self._conv2d(
            name=f"{conv_name_base}2b",
            num_filters=num_filters_2,
            kernel_size=kernel_size,
            stride=1,
            padding="same"
        )(x)
        x = self._batch_norm(f"{bn_name_base}2b")(x)
        x = tf.nn.relu(x)

        x = self._conv2d(
            name=f"{conv_name_base}2c",
            num_filters=num_filters_3,
            kernel_size=1,
            stride=1,
            padding="same"
        )(x)
        x = self._batch_norm(f"{bn_name_base}2c")(x)

        x = layers.Add()([x, shortcut])

        return tf.nn.relu(x)

    def _identity_block(
            self,
            inputs: tf.Tensor,
            stage: int,
            block: int,
            nums_filters: Tuple[int, int, int],
            kernel_size: int
    ) -> tf.Tensor:
        num_filters_1, num_filters_2, num_filters_3 = nums_filters

        conv_name_base = f"conv_stage{stage}_block{block}_branch"
        bn_name_base = f"bn_stage{stage}_block{block}_branch"

        x = self._conv2d(
            name=f"{conv_name_base}2a",
            num_filters=num_filters_1,
            kernel_size=1,
            stride=1,
            padding="same"
        )(inputs)
        x = self._batch_norm(f"{bn_name_base}2a")(x)
        x = tf.nn.relu(x)

        x = self._conv2d(
            name=f"{conv_name_base}2b",
            num_filters=num_filters_2,
            kernel_size=kernel_size,
            stride=1,
            padding="same"
        )(x)
        x = self._batch_norm(f"{bn_name_base}2b")(x)
        x = tf.nn.relu(x)

        x = self._conv2d(
            name=f"{conv_name_base}2c",
            num_filters=num_filters_3,
            kernel_size=1,
            stride=1,
            padding="same"
        )(x)
        x = self._batch_norm(f"{bn_name_base}2c")(x)

        x = layers.Add()([x, inputs])

        return tf.nn.relu(x)
