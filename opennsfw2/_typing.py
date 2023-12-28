"""
Typing utilities.
"""
from typing import TypeVar

import numpy as np
import numpy.typing

NDFloat32Array = np.typing.NDArray[np.float32]
NDUInt8Array = np.typing.NDArray[np.uint8]

# Note: Keras 3 has the KerasTensor class but not in Keras 2.
KerasTensor = TypeVar("KerasTensor")
