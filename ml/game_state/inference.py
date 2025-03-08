import os
import pathlib
from argparse import ArgumentParser, BooleanOptionalAction

import imageio
from matplotlib import animation
from matplotlib import pyplot as plt
from numpy import ndarray
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from torch import Tensor
from transformers import pipeline

from ml.game_state.videomae import get_datasets, get_label_id_dict, load_model


def run_inference(
    model_name: str,
    test_dataset: LabeledVideoDataset,
    show_gif: bool,
    gif_mean: float,
    gif_std: float,
    dataset_root_path: str,
    id2label: dict[int, str],
):
    video_classifier = pipeline(model=model_name, task="video-classification", device=0)

    for idx, sample_video in enumerate(test_dataset):
        print(f"Processing video {idx + 1} / {test_dataset.num_videos}")

        if show_gif:
            video_tensor = sample_video["video"]
            display_gif(video_tensor, gif_mean, gif_std)

        video_name = sample_video["video_name"]
        video_label = id2label[sample_video["label"]]
        video_path = os.path.join(dataset_root_path, "test", video_label, video_name)
        inference_result = video_classifier(video_path)

        success = inference_result[0]["label"] == video_label
        printColor = "\033[94m" if success else "\033[91m"
        print(
            f"{printColor}Inference result: {inference_result} on video {video_path} with actual label {video_label}"
        )


def denormalize_img(img: ndarray, mean: float, std: float):
    """
    de-normalizes the image pixels.
    """
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(
    video_tensor: Tensor, mean: float, std: float, filename: str = "sample.gif"
):
    """
    Prepares a GIF from a video tensor.
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = denormalize_img(
            video_frame.permute(1, 2, 0).numpy(), mean, std
        )
        frames.append(frame_unnormalized)

    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename


def display_gif(
    video_tensor: Tensor, mean: float, std: float, gif_name: str = "sample.gif"
):
    """
    Prepares and displays a GIF from a video tensor.
    """
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    gif_filename = create_gif(video_tensor, mean, std, gif_name)
    gif = imageio.mimread(gif_filename)
    fig, ax = plt.subplots()
    im = ax.imshow(gif[0])
    ax.axis("off")

    def update(frame):
        im.set_array(gif[frame])
        return [im]

    anim = animation.FuncAnimation(
        fig, update, frames=len(gif), interval=150, blit=True  # 150 ms between frames
    )

    plt.show()


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

    label2id, id2label = get_label_id_dict(pathlib.Path(args.test_root_path))
    image_processor, model = load_model(args.model_path, label2id, id2label)
    _, _, test_ds = get_datasets(image_processor, model, args.test_root_path)

    run_inference(
        model_name=args.model_path,
        test_dataset=test_ds,
        show_gif=args.demo,
        gif_mean=image_processor.image_mean,
        gif_std=image_processor.image_std,
        dataset_root_path=args.test_root_path,
        id2label=id2label,
    )
