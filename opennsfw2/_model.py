"""
Model architecture.

References:
https://github.com/mdietrichstein/tensorflow-open_nsfw
https://github.com/yahoo/open_nsfw
"""
import os
from typing import Optional, Tuple

import tensorflow as tf  # type: ignore
from tensorflow.keras import layers  # type: ignore # pylint: disable=import-error

from ._download import get_default_weights_path, download_weights_to


def _batch_norm(name: str) -> layers.BatchNormalization:
    return layers.BatchNormalization(
        name=name, epsilon=1e-05,  # Default used in Caffe.
    )


def _conv_block(
        stage: int,
        block: int,
        inputs: tf.Tensor,
        nums_filters: Tuple[int, int, int],
        kernel_size: int = 3,
        stride: int = 2,
) -> tf.Tensor:
    num_filters_1, num_filters_2, num_filters_3 = nums_filters

    conv_name_base = f"conv_stage{stage}_block{block}_branch"
    bn_name_base = f"bn_stage{stage}_block{block}_branch"
    shortcut_name_post = f"_stage{stage}_block{block}_proj_shortcut"
    final_activation_name = f"activation_stage{stage}_block{block}"
    activation_name_base = f"{final_activation_name}_branch"

    shortcut = layers.Conv2D(
        name=f"conv{shortcut_name_post}",
        filters=num_filters_3,
        kernel_size=1,
        strides=stride,
        padding="same"
    )(inputs)

    shortcut = _batch_norm(f"bn{shortcut_name_post}")(shortcut)

    x = layers.Conv2D(
        name=f"{conv_name_base}2a",
        filters=num_filters_1,
        kernel_size=1,
        strides=stride,
        padding="same"
    )(inputs)
    x = _batch_norm(f"{bn_name_base}2a")(x)
    x = layers.Activation("relu", name=f"{activation_name_base}2a")(x)

    x = layers.Conv2D(
        name=f"{conv_name_base}2b",
        filters=num_filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding="same"
    )(x)
    x = _batch_norm(f"{bn_name_base}2b")(x)
    x = layers.Activation("relu", name=f"{activation_name_base}2b")(x)

    x = layers.Conv2D(
        name=f"{conv_name_base}2c",
        filters=num_filters_3,
        kernel_size=1,
        strides=1,
        padding="same"
    )(x)
    x = _batch_norm(f"{bn_name_base}2c")(x)

    x = layers.Add()([x, shortcut])

    return layers.Activation("relu", name=final_activation_name)(x)


def _identity_block(
        stage: int,
        block: int,
        inputs: tf.Tensor,
        nums_filters: Tuple[int, int, int],
        kernel_size: int
) -> tf.Tensor:
    num_filters_1, num_filters_2, num_filters_3 = nums_filters

    conv_name_base = f"conv_stage{stage}_block{block}_branch"
    bn_name_base = f"bn_stage{stage}_block{block}_branch"
    final_activation_name = f"activation_stage{stage}_block{block}"
    activation_name_base = f"{final_activation_name}_branch"

    x = layers.Conv2D(
        name=f"{conv_name_base}2a",
        filters=num_filters_1,
        kernel_size=1,
        strides=1,
        padding="same"
    )(inputs)
    x = _batch_norm(f"{bn_name_base}2a")(x)
    x = layers.Activation("relu", name=f"{activation_name_base}2a")(x)

    x = layers.Conv2D(
        name=f"{conv_name_base}2b",
        filters=num_filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding="same"
    )(x)
    x = _batch_norm(f"{bn_name_base}2b")(x)
    x = layers.Activation("relu", name=f"{activation_name_base}2b")(x)

    x = layers.Conv2D(
        name=f"{conv_name_base}2c",
        filters=num_filters_3,
        kernel_size=1,
        strides=1,
        padding="same"
    )(x)
    x = _batch_norm(f"{bn_name_base}2c")(x)

    x = layers.Add()([x, inputs])

    return layers.Activation("relu", name=final_activation_name)(x)


def make_open_nsfw_model(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        weights_path: Optional[str] = get_default_weights_path()
) -> tf.keras.Model:
    image_input = layers.Input(shape=input_shape, name="input")
    x = image_input

    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
    x = layers.Conv2D(name="conv_1", filters=64, kernel_size=7, strides=2,
                      padding="valid")(x)

    x = _batch_norm("bn_1")(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = _conv_block(stage=0, block=0, inputs=x,
                    nums_filters=(32, 32, 128),
                    kernel_size=3, stride=1)

    x = _identity_block(stage=0, block=1, inputs=x,
                        nums_filters=(32, 32, 128), kernel_size=3)
    x = _identity_block(stage=0, block=2, inputs=x,
                        nums_filters=(32, 32, 128), kernel_size=3)

    x = _conv_block(stage=1, block=0, inputs=x,
                    nums_filters=(64, 64, 256),
                    kernel_size=3, stride=2)
    x = _identity_block(stage=1, block=1, inputs=x,
                        nums_filters=(64, 64, 256), kernel_size=3)
    x = _identity_block(stage=1, block=2, inputs=x,
                        nums_filters=(64, 64, 256), kernel_size=3)
    x = _identity_block(stage=1, block=3, inputs=x,
                        nums_filters=(64, 64, 256), kernel_size=3)

    x = _conv_block(stage=2, block=0, inputs=x,
                    nums_filters=(128, 128, 512),
                    kernel_size=3, stride=2)
    x = _identity_block(stage=2, block=1, inputs=x,
                        nums_filters=(128, 128, 512), kernel_size=3)
    x = _identity_block(stage=2, block=2, inputs=x,
                        nums_filters=(128, 128, 512), kernel_size=3)
    x = _identity_block(stage=2, block=3, inputs=x,
                        nums_filters=(128, 128, 512), kernel_size=3)
    x = _identity_block(stage=2, block=4, inputs=x,
                        nums_filters=(128, 128, 512), kernel_size=3)
    x = _identity_block(stage=2, block=5, inputs=x,
                        nums_filters=(128, 128, 512), kernel_size=3)

    x = _conv_block(stage=3, block=0, inputs=x,
                    nums_filters=(256, 256, 1024), kernel_size=3,
                    stride=2)
    x = _identity_block(stage=3, block=1, inputs=x,
                        nums_filters=(256, 256, 1024),
                        kernel_size=3)
    x = _identity_block(stage=3, block=2, inputs=x,
                        nums_filters=(256, 256, 1024),
                        kernel_size=3)

    x = layers.GlobalAveragePooling2D()(x)

    logits = layers.Dense(name="fc_nsfw", units=2)(x)
    output = layers.Activation("softmax", name="predictions")(logits)

    model = tf.keras.Model(image_input, output)

    if weights_path is not None:
        if not os.path.isfile(weights_path):
            download_weights_to(weights_path)
        model.load_weights(weights_path)
    return model
