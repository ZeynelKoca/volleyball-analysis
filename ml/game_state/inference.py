import gc
import os
import pathlib
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from gui.annotated_video import create_annotated_video, create_kpi_timeline
from gui.display_gif import display_gif
from ml.game_state.inference_result import InferenceResult
from ml.game_state.utils.video_utils import get_video_properties
from ml.game_state.videomae import get_datasets


def load_model(
    model_path: str,
) -> tuple[VideoMAEForVideoClassification, VideoMAEImageProcessor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {model_path}")

    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model_name_or_path=model_path, local_files_only=True
    )
    image_processor = VideoMAEImageProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        use_fast=True,
    )

    model = model.to(device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Number of classes: {model.config.num_labels}")

    return model, image_processor


def extract_frames(video_path: str, start_frame: int, num_frames: int) -> np.ndarray:
    """
    Extract a specific range of frames from a video file.

    Args:
        video_path: Path to the video file
        start_frame: Starting frame index
        num_frames: Number of frames to extract

    Returns:
        Numpy array of frames with shape [num_frames, height, width, channels]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if not frames:
        return np.array([])

    return np.array(frames)


def preprocess_frames(frames: np.ndarray, num_frames_required: int) -> torch.Tensor:
    """
    Preprocess a set of frames for the model.

    Args:
        frames: Numpy array of frames with shape [n, height, width, channels]
        num_frames_required: Number of frames required by the model

    Returns:
        Tensor of preprocessed frames
    """
    if len(frames) == 0:
        return None

    frames_tensor = torch.from_numpy(frames)

    # If the video is too long, sample frames uniformly
    if frames_tensor.shape[0] > num_frames_required:
        indices = torch.linspace(
            0, frames_tensor.shape[0] - 1, num_frames_required
        ).long()
        frames_tensor = frames_tensor[indices]
    # If the video is too short, pad with zeros
    elif frames_tensor.shape[0] < num_frames_required:
        print(
            f"Warning: Chunk has only {frames_tensor.shape[0]} frames, but model expects {num_frames_required}"
        )
        padding = torch.zeros(
            (num_frames_required - frames_tensor.shape[0], *frames_tensor.shape[1:]),
            dtype=frames_tensor.dtype,
        )
        frames_tensor = torch.cat([frames_tensor, padding], dim=0)

    return frames_tensor


def inference(
    model: VideoMAEForVideoClassification,
    pixel_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference with a VideoMAE model.
    """
    # Move input to the model's device
    pixel_values = pixel_values.to(model.device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        # Apply softmax to convert logits to probabilities
        probs = F.softmax(logits, dim=1).squeeze(0)

        # Get top-k
        k = min(model.config.num_labels, len(probs))
        topk = torch.topk(probs, k=k)

        confidences = topk.values.cpu().clone()
        indices = topk.indices.cpu().clone()

    return confidences, indices


def run_chunked_video_inference(
    model_path: str,
    video_path: str,
    chunk_duration: float = 2.0,  # Duration in seconds
    show_gif: bool = False,
) -> List[InferenceResult]:
    """
    Run inference on a video file by processing it in chunks of specified duration.
    """
    if not os.path.isfile(video_path):
        print(f"Error: video {video_path} is not a file")
        return []

    model, image_processor = load_model(model_path=model_path)

    # Use OpenCV to get video properties without loading the entire file
    total_frames, fps, dimensions, video_duration = get_video_properties(video_path)

    print(
        f"Video info: {total_frames} frames, {fps:.2f} FPS, {video_duration:.2f} seconds"
    )

    # Calculate frames per chunk
    frames_per_chunk = int(fps * chunk_duration)
    num_chunks = (
        total_frames + frames_per_chunk - 1
    ) // frames_per_chunk  # Ceiling division

    print(
        f"Processing in {num_chunks} chunks of {chunk_duration} seconds ({frames_per_chunk} frames per chunk)"
    )

    results = []

    for i in range(0, num_chunks):
        start_frame = i * frames_per_chunk
        end_frame = min((i + 1) * frames_per_chunk, total_frames)

        if start_frame >= total_frames:
            break

        start_time = start_frame / fps
        end_time = end_frame / fps

        print(
            f"\nProcessing chunk {i+1}/{num_chunks} (frames {start_frame}-{end_frame-1}, {start_time:.2f}s-{end_time:.2f}s)"
        )

        chunk_frames = extract_frames(video_path, start_frame, end_frame - start_frame)

        if len(chunk_frames) == 0:
            print(f"Warning: Could not extract frames for chunk {i+1}")
            continue

        processed_chunk = preprocess_frames(chunk_frames, model.config.num_frames)

        if processed_chunk is None:
            print(f"Warning: Failed to preprocess frames for chunk {i+1}")
            continue

        inputs = image_processor(
            list(processed_chunk.cpu().numpy()),
            return_tensors="pt",
        )

        confidences, indices = inference(
            model=model, pixel_values=inputs["pixel_values"]
        )

        timestamp = start_time
        result = InferenceResult.from_model_outputs(
            model=model,
            confidences=confidences,
            indices=indices,
            video_name=video_path,
            timestamp=timestamp,
        )
        results.append(result)

        result.print_results()

        if show_gif:
            try:
                video_tensor = inputs["pixel_values"].squeeze(0).permute(1, 0, 2, 3)
                display_gif(
                    video_tensor, image_processor.image_mean, image_processor.image_std
                )
            except Exception as e:
                print(f"Failed to display GIF: {e}")

        # Clean up memory
        del chunk_frames
        del processed_chunk
        del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 75)
    print(f"Video Analysis Summary for {video_path}")
    print("=" * 75)
    print(f"Processed {len(results)} chunks of {chunk_duration}s each")

    print("\nTimeline of detected game states:")
    for result in results:
        timestamp = result.timestamp
        state = result.top_prediction.label
        confidence = result.top_prediction.confidence
        print(f"{timestamp:6.2f}s - {state:15s} ({confidence*100:.1f}%)")

    return results


def run_video_inference(
    model_path: str, video_path: str, show_gif: bool = False
) -> InferenceResult:
    """
    Run inference on a single video file.
    """
    if not os.path.isfile(video_path):
        print(f"Error: video {video_path} is not a file")
        return None

    model, image_processor = load_model(model_path=model_path)

    total_frames, fps, dimensions, duration = get_video_properties(video_path)

    frames_to_extract = min(total_frames, model.config.num_frames * 3)

    # Sample frames uniformly across the video
    if total_frames > frames_to_extract:
        step = total_frames // frames_to_extract
        start_frame = (
            total_frames - (frames_to_extract * step)
        ) // 2  # Center the sampling
    else:
        step = 1
        start_frame = 0

    cap = cv2.VideoCapture(video_path)
    frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(frames_to_extract):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        # Skip frames according to step
        for _ in range(step - 1):
            cap.read()

    cap.release()

    if not frames:
        print(f"Error: Could not read frames from {video_path}")
        return None

    frames_np = np.array(frames)

    video = preprocess_frames(frames_np, model.config.num_frames)

    # Prepare inputs for the model
    inputs = image_processor(
        list(video.numpy()),
        return_tensors="pt",
    )

    confidences, indices = inference(model=model, pixel_values=inputs["pixel_values"])

    result = InferenceResult.from_model_outputs(
        model=model, confidences=confidences, indices=indices, video_name=video_path
    )
    result.print_results()

    if show_gif:
        try:
            video_tensor = inputs["pixel_values"].squeeze(0).permute(1, 0, 2, 3)
            display_gif(
                video_tensor, image_processor.image_mean, image_processor.image_std
            )
        except Exception as e:
            print(f"Failed to display GIF: {e}")

    return result


def run_dataset_inference(
    model_path: str,
    dataset_root_path: str,
    show_gif: bool = False,
) -> Dict[str, Any]:
    """
    Run inference on a dataset of videos.
    """
    model, image_processor = load_model(model_path=model_path)

    _, _, test_dataset = get_datasets(
        image_processor, model, pathlib.Path(dataset_root_path), model.config.label2id
    )

    stats = {"success": 0, "skipped": 0, "failed": 0, "total": test_dataset.num_videos}
    results = []

    for idx, sample_video in enumerate(test_dataset):
        print(f"Processing video {idx + 1} / {test_dataset.num_videos}")

        video_tensor = sample_video["video"].permute(1, 0, 2, 3)
        video_name = sample_video["video_name"]
        true_label = model.config.id2label[sample_video["label"]]

        pixel_values = video_tensor.unsqueeze(0)
        confidences, indices = inference(model, pixel_values)

        result = InferenceResult.from_model_outputs(
            model=model,
            confidences=confidences,
            indices=indices,
            video_name=video_name,
            true_label=true_label,
        )

        results.append(result)

        # Check if the result is confident enough
        is_valid = result.top_prediction and result.top_prediction.confidence > 0.8

        if not is_valid:
            stats["skipped"] += 1
        elif result.is_correct:
            stats["success"] += 1
        else:
            stats["failed"] += 1

        result.print_results()

        if show_gif:
            display_gif(
                video_tensor, image_processor.image_mean, image_processor.image_std
            )

        # Clean up memory
        del video_tensor
        del pixel_values
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary statistics
    valid_count = stats["total"] - stats["skipped"]
    success_rate = (stats["success"] / valid_count * 100) if valid_count > 0 else 0
    failure_rate = (stats["failed"] / valid_count * 100) if valid_count > 0 else 0

    print("\nInference statistics:")
    print(f"Total videos: {stats['total']}")
    print(f"Skipped (low confidence): {stats['skipped']} videos")
    print(
        f"Successfully classified: {stats['success']} / {valid_count} ({success_rate:.1f}%)"
    )
    print(
        f"Incorrectly classified: {stats['failed']} / {valid_count} ({failure_rate:.1f}%)"
    )

    return {"stats": stats, "results": [result.to_dict() for result in results]}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-path",
        "-M",
        type=str,
        default="videomae-base-finetuned-ucf101-subset/checkpoint-740",
        help="The path to the pre-trained model to be used for inference",
    )
    parser.add_argument(
        "--gif-demo",
        "-G",
        type=bool,
        action=BooleanOptionalAction,
        help="Whether a demo gif should be shown for the item(s) being inferenced",
    )
    parser.add_argument(
        "--dataset-path",
        "-D",
        type=str,
        help="The directory (residing in project root) that contains the custom dataset. Directory should consist of sub-folders with label names, where video data resides",
    )
    parser.add_argument(
        "--video-path",
        "-V",
        type=str,
        help="A singular video file path (mp4, avi, mov) to inference. Will be ignored if --dataset-path is set",
    )
    parser.add_argument(
        "--chunked",
        "-C",
        type=bool,
        action=BooleanOptionalAction,
        help="Process video in chunks of specified duration",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=2.0,
        help="Duration of each chunk in seconds for chunked processing",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        action=BooleanOptionalAction,
        help="Visualize the input video including labeled chunks",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset_path is not None:
        results = run_dataset_inference(
            model_path=args.model_path,
            dataset_root_path=args.dataset_path,
            show_gif=args.gif_demo,
        )
        print(f"Completed inference on {len(results['results'])} videos")
    elif args.video_path is not None:
        if args.chunked:
            results = run_chunked_video_inference(
                model_path=args.model_path,
                video_path=args.video_path,
                chunk_duration=args.chunk_duration,
                show_gif=args.gif_demo,
            )
            print(f"Completed chunked inference with {len(results)} segments")

            if args.visualize and results:
                timeline_image = create_kpi_timeline(
                    results=results, video_name=os.path.basename(args.video_path)
                )

                annotated_video = create_annotated_video(
                    video_path=args.video_path, results=results
                )

                print(f"Created visualizations: {annotated_video}, {timeline_image}")
        else:
            result = run_video_inference(
                model_path=args.model_path,
                video_path=args.video_path,
                show_gif=args.gif_demo,
            )
            print("Single video inference complete")
    else:
        print("Error: Either --dataset-path or --video-path must be provided")
