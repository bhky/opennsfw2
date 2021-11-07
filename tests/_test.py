"""
Unit tests.
"""

import os
import unittest

import numpy as np

import opennsfw2 as n2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = n2.make_open_nsfw_model()
IMAGE_PATHS = [
    os.path.join(BASE_DIR, "test_image_1.jpg"),
    os.path.join(BASE_DIR, "test_image_2.jpg"),
    os.path.join(BASE_DIR, "test_image_3.jpg"),
]


class TestModel(unittest.TestCase):

    def _assert(self, expected: np.ndarray, predicted: np.ndarray) -> None:
        for expected_score, predicted_score in zip(expected, predicted):
            self.assertAlmostEqual(expected_score, predicted_score, places=3)

    def test_inference_yahoo_preprocessing(self) -> None:
        expected_scores = np.array([0.012, 0.756, 0.067])
        predicted_scores = n2.predict(
            IMAGE_PATHS, preprocessing=n2.Preprocessing.YAHOO
        )[:, 1]
        self._assert(expected_scores, predicted_scores)

    def test_inference_simple_preprocessing(self) -> None:
        expected_scores = np.array([0.001, 0.597, 0.003])
        predicted_scores = n2.predict(
            IMAGE_PATHS, preprocessing=n2.Preprocessing.SIMPLE
        )[:, 1]
        self._assert(expected_scores, predicted_scores)
