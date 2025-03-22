import pathlib
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_video
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from ml.game_state.utils.display_gif import display_gif
from ml.game_state.videomae import get_datasets


class PredictionItem:
    """
    Represents a single prediction with label and confidence.
    """

    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence

    def __str__(self) -> str:
        return f"{self.label}: {self.confidence:.4f} ({self.confidence*100:.2f}%)"


class InferenceResult:
    """
    Class to encapsulate the results of model inference on a video.
    """

    def __init__(
        self,
        predictions: List[PredictionItem],
        video_name: Optional[str] = None,
        true_label: Optional[str] = None,
    ):
        self.predictions = predictions
        self.video_name = video_name
        self.true_label = true_label

        # Sort predictions by confidence (highest first)
        self.predictions.sort(key=lambda x: x.confidence, reverse=True)

    @property
    def top_prediction(self) -> PredictionItem:
        """Returns the prediction with highest confidence."""
        return self.predictions[0]

    @property
    def is_correct(self) -> Optional[bool]:
        """Returns whether the top prediction matches the true label, if available."""
        if self.true_label is None:
            return None
        return self.top_prediction.label == self.true_label

    def print_results(self, show_correctness: bool = True):
        """Print inference results in a formatted way."""
        file_info = f" for {self.video_name}" if self.video_name else ""
        print(f"\nInference results{file_info}:")
        print("-" * (20 + len(file_info)))

        for idx, pred in enumerate(self.predictions):
            print(f"{idx+1}. {pred}")

        print(f"\nTop prediction: {self.top_prediction}")

        if show_correctness and self.true_label:
            status = "\033[92mCORRECT" if self.is_correct else "\033[91mINCORRECT"
            print(f"{status} prediction. True label: {self.true_label}\033[0m\n")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary representation."""
        return {
            "video_name": self.video_name,
            "top_prediction": {
                "label": self.top_prediction.label,
                "confidence": self.top_prediction.confidence,
            },
            "predictions": [
                {"label": p.label, "confidence": p.confidence} for p in self.predictions
            ],
            "true_label": self.true_label,
            "is_correct": self.is_correct,
        }

    @classmethod
    def from_model_outputs(
        cls,
        model: VideoMAEForVideoClassification,
        confidences: torch.Tensor,
        indices: torch.Tensor,
        video_name: Optional[str] = None,
        true_label: Optional[str] = None,
    ) -> "InferenceResult":
        """
        Create an InferenceResult from model outputs.
        """
        # Convert tensors to CPU numpy arrays if they're not already
        confidences_np = confidences.cpu().numpy()
        indices_np = indices.cpu().numpy()

        # Create prediction items
        predictions = [
            PredictionItem(
                label=model.config.id2label[int(idx)], confidence=float(conf)
            )
            for idx, conf in zip(indices_np, confidences_np)
        ]

        return cls(predictions, video_name, true_label)


def load_model(
    model_path: str,
) -> tuple[VideoMAEForVideoClassification, VideoMAEImageProcessor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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


def preprocess_video(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Preprocess a video tensor to have the expected number of frames.
    """
    # If the video is too long, sample frames uniformly
    if video.shape[0] > num_frames:
        indices = torch.linspace(0, video.shape[0] - 1, num_frames).long()
        video = video[indices]
    # If the video is too short, pad with zeros
    elif video.shape[0] < num_frames:
        print(
            f"Warning: Video has only {video.shape[0]} frames, but model expects {num_frames}"
        )
        padding = torch.zeros(
            (num_frames - video.shape[0], *video.shape[1:]), dtype=video.dtype
        )
        video = torch.cat([video, padding], dim=0)

    return video


def run_inference(
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
        confidences = (
            topk.values.cuda().clone()
            if torch.cuda.is_available()
            else topk.values.cpu().clone()
        )
        indices = (
            topk.indices.cuda().clone()
            if torch.cuda.is_available()
            else topk.indices.cpu().clone()
        )

    return confidences, indices


def format_results(
    model: VideoMAEForVideoClassification,
    confidences: torch.Tensor,
    indices: torch.Tensor,
) -> Dict[str, Any]:
    """
    Format inference results into a structured dictionary.
    """
    # Convert tensors to CPU numpy arrays if they're not already
    confidences_np = confidences.cpu().numpy()
    indices_np = indices.cpu().numpy()

    # Format results
    return {
        "top_prediction": model.config.id2label[indices_np[0]],
        "confidence": float(confidences_np[0]),
        "all_predictions": [
            {"label": model.config.id2label[int(idx)], "confidence": float(conf)}
            for idx, conf in zip(indices_np, confidences_np)
        ],
    }


def run_video_inference(
    model_path: str, video_path: str, show_gif: bool = False
) -> InferenceResult:
    """
    Run inference on a single video file.
    """
    if not pathlib.Path(video_path).is_file():
        print(f"Error: video {video_path} is not a file")
        return

    print(f"Loading model from {model_path}")
    model, image_processor = load_model(model_path=model_path)

    video, _, _ = read_video(video_path, pts_unit="sec")
    video = preprocess_video(video, model.config.num_frames)

    # Prepare inputs for the model
    inputs = image_processor(
        list(video.numpy()),  # Convert tensor to numpy arrays
        return_tensors="pt",
    )

    confidences, indices = run_inference(
        model=model, pixel_values=inputs["pixel_values"]
    )

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
        confidences, indices = run_inference(model, pixel_values)

        result = InferenceResult.from_model_outputs(
            model=model,
            confidences=confidences,
            indices=indices,
            video_name=video_name,
            true_label=true_label,
        )

        results.append(result)

        # Check if the result is confident enough
        is_valid = result.top_prediction.confidence > 0.8

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
        result = run_video_inference(
            model_path=args.model_path,
            video_path=args.video_path,
            show_gif=args.gif_demo,
        )
        print("Single video inference complete")
    else:
        print("Error: Either --dataset-path or --video-path must be provided")
