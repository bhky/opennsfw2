"""
Keras compatibility layer supporting Keras 3 and tf-keras.
"""
from typing import Any, Tuple

try:
    import keras  # type: ignore
    from keras import layers, Model  # type: ignore
    from keras import backend as keras_backend  # type: ignore
    from keras.preprocessing.image import array_to_img  # type: ignore  # pylint: disable=import-error
    _TF_KERAS = False
except ImportError:
    import tf_keras as keras  # type: ignore
    from tf_keras import layers, Model  # type: ignore
    from tf_keras import backend as keras_backend  # type: ignore
    from tf_keras.preprocessing.image import array_to_img  # type: ignore
    _TF_KERAS = True


def ops_cast(x: Any, dtype: str) -> Any:
    if not _TF_KERAS:
        return keras.ops.cast(x, dtype)
    import tensorflow as tf  # type: ignore  # pylint: disable=import-outside-toplevel
    return tf.cast(x, dtype)


def ops_image_resize(
        x: Any, size: Tuple[int, int], interpolation: str = "bilinear"
) -> Any:
    if not _TF_KERAS:
        return keras.ops.image.resize(x, size, interpolation=interpolation)
    import tensorflow as tf  # type: ignore  # pylint: disable=import-outside-toplevel
    return tf.image.resize(x, size, method=interpolation)