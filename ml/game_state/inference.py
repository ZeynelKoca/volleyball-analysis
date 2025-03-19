import os
import pathlib
from argparse import ArgumentParser, BooleanOptionalAction

from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from transformers import (VideoMAEForVideoClassification,
                          VideoMAEImageProcessor, pipeline)

from ml.game_state.utils.display_gif import display_gif
from ml.game_state.videomae import get_datasets, get_label_id_dict, load_model


def run_inference(
    model_path: str,
    test_dataset: LabeledVideoDataset,
    show_gif: bool,
    gif_mean: float,
    gif_std: float,
    testset_root_path: str,
    id2label: dict[int, str],
):
    model = VideoMAEForVideoClassification.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
    image_processor = VideoMAEImageProcessor.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)
    video_classifier = pipeline(model=model, task="video-classification", device=0, use_fast=True, image_processor=image_processor)

    for idx, sample_video in enumerate(test_dataset):
        print(f"Processing video {idx + 1} / {test_dataset.num_videos}")

        video_tensor = sample_video["video"].permute(1, 0, 2, 3)

        if show_gif:
            display_gif(video_tensor, gif_mean, gif_std)

        video_name = sample_video["video_name"]
        video_label = id2label[sample_video["label"]]
        video_path = os.path.join(testset_root_path, "test", video_label, video_name)
        inference_result = video_classifier(video_path)

        success = inference_result[0]["label"] == video_label
        printColor = "\033[94m" if success else "\033[91m"
        print(
            f"{printColor}Inference result: {inference_result} on video {video_path} with actual label {video_label}'\033[0m'"
        )


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
        "--demo",
        "-D",
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

    label2id, id2label = get_label_id_dict(pathlib.Path(args.test_root_path), "mp4")
    #label2id = {"no-game": 1, "rally":2, "serve":3}
    #id2label = {1: "no-game", 2: "rally", 3: "serve"}
    image_processor, model = load_model(args.model_path, label2id, id2label)
    _, _, test_ds = get_datasets(image_processor, model, args.test_root_path)

    run_inference(
        model_path=args.model_path,
        test_dataset=test_ds,
        show_gif=args.demo,
        gif_mean=image_processor.image_mean,
        gif_std=image_processor.image_std,
        testset_root_path=args.test_root_path,
        id2label=id2label,
    )
