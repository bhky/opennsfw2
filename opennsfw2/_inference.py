"""
Inference utilities.
"""

from typing import List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

from ._image import preprocess_image, Preprocessing
from ._model import get_default_weights_path, make_open_nsfw_model


def predict_images(
        image_paths: Sequence[str],
        batch_size: int = 16,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path()
) -> List[float]:
    """
    Pipeline from image paths to predicted NSFW probabilities.
    """
    images = np.array([
        preprocess_image(Image.open(image_path), preprocessing)
        for image_path in image_paths
    ])
    model = make_open_nsfw_model(weights_path=weights_path)
    predictions = model.predict(images, batch_size=batch_size)
    nsfw_probabilities: List[float] = predictions[:, 1].tolist()
    return nsfw_probabilities


def predict_image(
        image_path: str,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path()
) -> float:
    """
    Pipeline from single image path to predicted NSFW probability.
    """
    return predict_images(
        [image_path],
        batch_size=1,
        preprocessing=preprocessing,
        weights_path=weights_path
    )[0]


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
    cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
    fps = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=no-member

    model = make_open_nsfw_model(weights_path=weights_path)

    video_writer: Optional[cv2.VideoWriter] = None  # pylint: disable=no-member
    nsfw_probability = 0.0
    nsfw_probabilities: List[float] = []
    frame_count = 0

    while cap.isOpened():
        ret, bgr_frame = cap.read()  # Get next video frame.
        if not ret:
            break  # End of given video.

        frame_count += 1
        frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

        if video_writer is None and output_video_path is not None:
            video_writer = cv2.VideoWriter(  # pylint: disable=no-member
                output_video_path,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # pylint: disable=no-member
                fps, (frame.shape[1], frame.shape[0])
            )

        if frame_count == 1 or (frame_count + 1) % frame_interval == 0:
            pil_frame = Image.fromarray(frame)
            input_frame = preprocess_image(pil_frame, preprocessing)
            predictions = model.predict(np.expand_dims(input_frame, axis=0), 1)
            nsfw_probability = np.round(predictions[0][1], 2)

        nsfw_probabilities.append(nsfw_probability)

        if video_writer is not None:
            result_text = f"NSFW probability: {str(nsfw_probability)}"
            # RGB colour.
            colour = (255, 0, 0) if nsfw_probability >= 0.8 else (0, 0, 255)
            cv2.putText(  # pylint: disable=no-member
                frame, result_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
                1, colour, 2, cv2.LINE_AA  # pylint: disable=no-member
            )
            video_writer.write(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
            )

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()  # pylint: disable=no-member

    elapsed_seconds = (np.arange(1, len(nsfw_probabilities) + 1) / fps).tolist()
    return elapsed_seconds, nsfw_probabilities
