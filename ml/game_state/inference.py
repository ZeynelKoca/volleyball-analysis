import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from ml.game_state.utils.display_gif import display_gif
from ml.game_state.videomae import get_datasets


def run_inference(
    model_path: str,
    show_gif: bool,
    testset_root_path: str,
):

    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model_name_or_path=model_path, local_files_only=True
    )
    image_processor = VideoMAEImageProcessor.from_pretrained(
        pretrained_model_name_or_path=model_path, local_files_only=True
    )
    _, _, test_dataset = get_datasets(image_processor, model, testset_root_path)

    for idx, sample_video in enumerate(test_dataset):
        print(f"Processing video {idx + 1} / {test_dataset.num_videos}")

        video_tensor = sample_video["video"].permute(1, 0, 2, 3)

        video_name = sample_video["video_name"]
        video_label = model.config.id2label[sample_video["label"]]
        video_path = os.path.join(testset_root_path, "test", video_label, video_name)
        predicted_confidences, predicted_labels = __inference(
            model=model, sample_video=sample_video
        )

        inference_result = [
            {
                "prediction": model.config.id2label[predicted_labels[idx].item()],
                "confidence": predicted_confidences[idx].item(),
            }
            for idx in range(0, model.config.num_labels)
        ]
        success = inference_result[0]["prediction"] == video_label
        printColor = "\033[94m" if success else "\033[91m"

        if (
            abs(inference_result[0]["confidence"] - inference_result[1]["confidence"])
            < 0.20
        ):
            # Deem the inference invalid if the top 2 results are too close in confidence
            success = False
            print(
                f"\033[93m Could not confidently inference video {video_path} with actual label {video_label}. {printColor}Inference result was {inference_result}"
            )
        else:
            print(
                f"{printColor}Inference result: {inference_result} on video {video_path} with actual label {video_label}'\033[0m"
            )

        if show_gif:
            gif_mean = image_processor.image_mean
            gif_std = image_processor.image_std
            display_gif(video_tensor, gif_mean, gif_std)


def __inference(model, sample_video):
    """Utility to run inference given a model and test video.
    The video is assumed to be preprocessed already.
    """
    # (num_frames, num_channels, height, width)
    permuted_sample_test_video = sample_video["video"].permute(1, 0, 2, 3)
    inputs = {
        "pixel_values": permuted_sample_test_video.unsqueeze(0),
        "labels": torch.tensor(
            [sample_video["label"]]
        ),  # this can be skipped if you don't have labels available.
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to convert logits to probabilities
        results = F.softmax(logits, dim=1).squeeze(0)

    # Return all predicted label indices and respective confidences, ordered by highest confidence
    return torch.topk(results, k=min(model.config.num_labels, len(results)))


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
        "--test-root-path",
        "-T",
        type=str,
        help="The path to the directory containing all videos to be inferenced",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_inference(
        model_path=args.model_path,
        show_gif=args.demo,
        testset_root_path=args.test_root_path,
    )
