"""
Inference utilities.
"""
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore

from keras import Model  # type: ignore

from ._download import get_default_weights_path
from ._image import preprocess_image, Preprocessing
from ._model import make_open_nsfw_model
from ._typing import NDFloat32Array

global_model: Optional[Model] = None
global_model_path: Optional[str] = None


def _update_global_model_if_needed(weights_path: Optional[str]) -> None:
    global global_model
    global global_model_path
    if global_model is None or weights_path != global_model_path:
        global_model = make_open_nsfw_model(weights_path=weights_path)
        global_model_path = weights_path


def predict_image(
        image_path: str,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path(),
        grad_cam_path: Optional[str] = None,
        alpha: float = 0.8
) -> float:
    """
    Pipeline from single image path to predicted NSFW probability.
    Optionally generate and save the Grad-CAM plot.
    """
    pil_image = Image.open(image_path)
    image = preprocess_image(pil_image, preprocessing)
    _update_global_model_if_needed(weights_path)
    assert global_model is not None
    nsfw_probability = float(global_model(np.expand_dims(image, 0))[0][1])

    if grad_cam_path is not None:
        # TensorFlow will only be imported here.
        from ._inspection import make_and_save_nsfw_grad_cam

        make_and_save_nsfw_grad_cam(
            pil_image, preprocessing, global_model, grad_cam_path, alpha
        )

    return nsfw_probability


def _predict_from_image_paths_in_batches(
        model_: Model,
        image_paths: Sequence[str],
        batch_size: int,
        preprocessing: Preprocessing
) -> NDFloat32Array:
    """
    It is on purpose not to use `model.predict` because many users would like
    to use the API in loops. See here:
    https://keras.io/api/models/model_training_apis/#predict-method
    """
    prediction_batches: List[Any] = []
    for i in range(0, len(image_paths), batch_size):
        path_batch = image_paths[i: i + batch_size]
        image_batch = [
            preprocess_image(Image.open(path), preprocessing)
            for path in path_batch
        ]
        prediction_batches.append(model_(np.array(image_batch)))
    predictions: NDFloat32Array = np.concatenate(prediction_batches, axis=0)
    return predictions


def predict_images(
        image_paths: Sequence[str],
        batch_size: int = 8,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path(),
        grad_cam_paths: Optional[Sequence[str]] = None,
        alpha: float = 0.8
) -> List[float]:
    """
    Pipeline from image paths to predicted NSFW probabilities.
    Optionally generate and save the Grad-CAM plots.
    """
    _update_global_model_if_needed(weights_path)
    predictions = _predict_from_image_paths_in_batches(
        global_model, image_paths, batch_size, preprocessing
    )
    nsfw_probabilities: List[float] = predictions[:, 1].tolist()

    if grad_cam_paths is not None:
        # TensorFlow will only be imported here.
        from ._inspection import make_and_save_nsfw_grad_cam

        for image_path, grad_cam_path in zip(image_paths, grad_cam_paths):
            make_and_save_nsfw_grad_cam(
                Image.open(image_path), preprocessing, global_model,
                grad_cam_path, alpha
            )

    return nsfw_probabilities


class Aggregation(str, Enum):
    MEAN = auto()
    MEDIAN = auto()
    MAX = auto()
    MIN = auto()


def _get_aggregation_fn(
        aggregation: Aggregation
) -> Callable[[NDFloat32Array], float]:

    def fn(x: NDFloat32Array) -> float:
        agg: Any = {
            Aggregation.MEAN: np.mean,
            Aggregation.MEDIAN: np.median,
            Aggregation.MAX: np.max,
            Aggregation.MIN: np.min,
        }[aggregation]
        return float(agg(x))

    return fn


def _predict_preprocessed_images_in_batches(
        model_: Model,
        images: List[NDFloat32Array],
        batch_size: int
) -> NDFloat32Array:
    prediction_batches: List[Any] = []
    for i in range(0, len(images), batch_size):
        batch = np.array(images[i: i + batch_size])
        prediction_batches.append(model_(batch))
    predictions: NDFloat32Array = np.concatenate(prediction_batches, axis=0)
    return predictions


def predict_video_frames(
        video_path: str,
        frame_interval: int = 8,
        aggregation_size: int = 8,
        aggregation: Aggregation = Aggregation.MEAN,
        batch_size: int = 8,
        output_video_path: Optional[str] = None,
        preprocessing: Preprocessing = Preprocessing.YAHOO,
        weights_path: Optional[str] = get_default_weights_path(),
        progress_bar: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Make prediction for each video frame.
    """
    cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
    fps = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=no-member

    _update_global_model_if_needed(weights_path)

    video_writer: Optional[cv2.VideoWriter] = None  # pylint: disable=no-member
    input_frames: List[NDFloat32Array] = []
    nsfw_probability = 0.0
    nsfw_probabilities: List[float] = []
    frame_count = 0

    if progress_bar:
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))  # pylint: disable=no-member
    else:
        pbar = None

    while cap.isOpened():
        ret, bgr_frame = cap.read()  # Get next video frame.
        if not ret:
            break  # End of given video.

        if pbar is not None:
            pbar.update(1)

        frame_count += 1
        frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

        if video_writer is None and output_video_path is not None:
            video_writer = cv2.VideoWriter(  # pylint: disable=no-member
                output_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore # pylint: disable=no-member
                fps, (frame.shape[1], frame.shape[0])
            )

        if frame_count == 1 or (frame_count + 1) % frame_interval == 0:
            pil_frame = Image.fromarray(frame)
            input_frame = preprocess_image(pil_frame, preprocessing)
            input_frames.append(input_frame)

            if frame_count == 1 or len(input_frames) >= aggregation_size:
                predictions = _predict_preprocessed_images_in_batches(
                    global_model, input_frames, batch_size
                )
                agg_fn = _get_aggregation_fn(aggregation)
                nsfw_probability = agg_fn(predictions[:, 1])
                input_frames = []

        nsfw_probabilities.append(nsfw_probability)

        if video_writer is not None:
            prob_str = str(np.round(nsfw_probability, 2))
            result_text = f"NSFW probability: {prob_str}"
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

    if pbar is not None:
        pbar.close()

    elapsed_seconds = (np.arange(1, len(nsfw_probabilities) + 1) / fps).tolist()
    return elapsed_seconds, nsfw_probabilities
