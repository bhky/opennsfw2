"""
Unit test.
"""
import os
import unittest
from typing import Optional, Sequence

from keras import backend as keras_backend

import opennsfw2 as n2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATHS = [
    os.path.join(BASE_DIR, "test_image_1.jpg"),
    os.path.join(BASE_DIR, "test_image_2.jpg"),
    os.path.join(BASE_DIR, "test_image_3.jpg"),
]
OUTPUT_GRAD_CAM_PATHS = [  # Only used with TensorFlow backend.
    os.path.join(BASE_DIR, "output_grad_cam_1.jpg"),
    os.path.join(BASE_DIR, "output_grad_cam_2.jpg"),
    os.path.join(BASE_DIR, "output_grad_cam_3.jpg"),
]
VIDEO_PATH = os.path.join(BASE_DIR, "test_video.mp4")


class TestModel(unittest.TestCase):

    def _assert(
            self,
            expected: Sequence[float],
            predicted: Sequence[float],
            paths: Optional[Sequence[str]] = None
    ) -> None:
        for i, (expected_p, predicted_p) in enumerate(zip(expected, predicted)):
            self.assertAlmostEqual(expected_p, predicted_p, places=3)
            if paths:
                self.assertTrue(os.path.exists(paths[i]))

    def test_predict_images_yahoo_preprocessing(self) -> None:
        if keras_backend.backend() == "tensorflow":
            grad_cam_paths = OUTPUT_GRAD_CAM_PATHS
        else:
            grad_cam_paths = None

        expected_probabilities = [0.016, 0.983, 0.077]
        predicted_probabilities = n2.predict_images(
            IMAGE_PATHS,
            preprocessing=n2.Preprocessing.YAHOO,
            grad_cam_paths=grad_cam_paths
        )
        self._assert(
            expected_probabilities, predicted_probabilities,
            OUTPUT_GRAD_CAM_PATHS
        )

    def test_predict_images_simple_preprocessing(self) -> None:
        expected_probabilities = [0.001, 0.913, 0.003]
        predicted_probabilities = n2.predict_images(
            IMAGE_PATHS, preprocessing=n2.Preprocessing.SIMPLE
        )
        self._assert(expected_probabilities, predicted_probabilities)

    def test_predict_image(self) -> None:
        self.assertAlmostEqual(
            0.983, n2.predict_image(IMAGE_PATHS[1]), places=3
        )

    def test_predict_video_frames(self) -> None:
        elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(
            VIDEO_PATH, frame_interval=2, progress_bar=False
        )
        self.assertGreater(len(elapsed_seconds), 0)
        self.assertEqual(len(elapsed_seconds), len(nsfw_probabilities))
