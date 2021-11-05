"""
Unit tests.
"""

import unittest

import numpy as np

import opennsfw2 as n2


MODEL = n2.make_open_nsfw_model()


class TestModel(unittest.TestCase):

    def test_inference_simple_preprocessing(self) -> None:
        images = np.array([
            n2.load_and_preprocess_image(
                "tests/test_image_1.jpg", n2.Preprocessing.SIMPLE
            ),
            n2.load_and_preprocess_image(
                "tests/test_image_2.jpg", n2.Preprocessing.SIMPLE
            ),
        ])
        expected_scores = [0.001, 0.597]

        predictions = MODEL.predict(images)
        for expected_score, prediction in zip(expected_scores, predictions):
            score = prediction[1]
            self.assertAlmostEqual(expected_score, score, places=3)

    def test_inference_yahoo_preprocessing(self) -> None:
        images = np.array([
            n2.load_and_preprocess_image(
                "tests/test_image_1.jpg", n2.Preprocessing.YAHOO
            ),
            n2.load_and_preprocess_image(
                "tests/test_image_2.jpg", n2.Preprocessing.YAHOO
            ),
        ])
        expected_scores = [0.012, 0.756]

        predictions = MODEL.predict(images)
        for expected_score, prediction in zip(expected_scores, predictions):
            score = prediction[1]
            self.assertAlmostEqual(expected_score, score, places=3)
