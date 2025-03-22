from typing import Any, Dict, List, Optional

import torch
from transformers import VideoMAEForVideoClassification


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
        timestamp: Optional[float] = None,
    ):
        self.predictions = predictions
        self.video_name = video_name
        self.true_label = true_label
        self.timestamp = timestamp

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
        timestamp_info = (
            f" at {self.timestamp:.2f}s" if self.timestamp is not None else ""
        )
        print(f"\nInference results{file_info}{timestamp_info}:")
        print("-" * (20 + len(file_info) + len(timestamp_info)))

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
            "timestamp": self.timestamp,
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
        timestamp: Optional[float] = None,
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

        return cls(predictions, video_name, true_label, timestamp)
