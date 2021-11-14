"""
Unit tests.
"""

import os
import unittest
from typing import List

import opennsfw2 as n2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = n2.make_open_nsfw_model()
IMAGE_PATHS = [
    os.path.join(BASE_DIR, "test_image_1.jpg"),
    os.path.join(BASE_DIR, "test_image_2.jpg"),
    os.path.join(BASE_DIR, "test_image_3.jpg"),
]
VIDEO_PATH = os.path.join(BASE_DIR, "test_video.mp4")


class TestModel(unittest.TestCase):

    def _assert(self, expected: List[float], predicted: List[float]) -> None:
        for expected_score, predicted_score in zip(expected, predicted):
            self.assertAlmostEqual(expected_score, predicted_score, places=3)

    def test_predict_images_yahoo_preprocessing(self) -> None:
        expected_scores = [0.012, 0.756, 0.067]
        predicted_scores = n2.predict_images(
            IMAGE_PATHS, preprocessing=n2.Preprocessing.YAHOO
        )
        self._assert(expected_scores, predicted_scores)

    def test_predict_images_simple_preprocessing(self) -> None:
        expected_scores = [0.001, 0.597, 0.003]
        predicted_scores = n2.predict_images(
            IMAGE_PATHS, preprocessing=n2.Preprocessing.SIMPLE
        )
        self._assert(expected_scores, predicted_scores)

    def test_predict_image(self) -> None:
        self.assertAlmostEqual(
            0.756, n2.predict_image(IMAGE_PATHS[1]), places=3
        )

    def test_predict_video_frames(self) -> None:
        elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(
            VIDEO_PATH, frame_interval=10
        )
        self.assertGreater(len(elapsed_seconds), 0)
        self.assertEqual(len(elapsed_seconds), len(nsfw_probabilities))
