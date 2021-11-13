"""
Inference utilities.
"""

import os
from typing import List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

from ._image import preprocess_image, Preprocessing
from ._model import get_default_weights_path, make_open_nsfw_model


def predict(
        image_paths: Sequence[str],
        batch_size: int = 32,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path()
) -> np.ndarray:
    """
    Pipeline from image paths to predictions.
    """
    images = np.array([
        preprocess_image(Image.open(image_path), preprocessing)
        for image_path in image_paths
    ])
    model = make_open_nsfw_model(weights_path=weights_path)
    return model.predict(images, batch_size=batch_size)


def predict_video_frames(
        video_path: str,
        frame_interval: int = 8,
        output_video_path: Optional[str] = None,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path()
) -> Tuple[List[float], List[float]]:
    """
    Make prediction for each video frame.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model = make_open_nsfw_model(weights_path=weights_path)

    video_writer: Optional[cv2.VideoWriter] = None
    nsfw_probability = 0.0
    nsfw_probabilities: List[float] = []
    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()  # Get next video frame.
        if not ret:
            break  # End of given video.

        frame_count += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if video_writer is None and output_video_path is not None:
            video_writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                fps, (frame.shape[1], frame.shape[0])
            )

        if frame_count == 1 or (frame_count + 1) % frame_interval == 0:
            pil_frame = Image.fromarray(frame)
            input_frame = preprocess_image(pil_frame, preprocessing)
            predictions = model.predict(np.expand_dims(input_frame, axis=0), 1)
            nsfw_probability = predictions[0][1]

        nsfw_probabilities.append(nsfw_probability)

        result_text = f"NSFW probability: {str(np.round(nsfw_probability, 2))}"
        cv2.putText(
            frame, result_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2, cv2.LINE_AA
        )

        if video_writer is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

    seconds = (np.arange(1, len(nsfw_probabilities) + 1) / fps).tolist()
    return seconds, nsfw_probabilities
