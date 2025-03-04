from enum import Enum
import imageio
import os
import pathlib
import tarfile
from matplotlib import animation
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from huggingface_hub import hf_hub_download
from numpy import ndarray
from torch import Tensor
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from definitions import ROOT_DIR
import pytorchvideo.data
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


class Dataset(Enum):
    TRAIN = "train"
    VALIDATE = "val"
    EVALUATE = "test"


def use_pretrained_hf_model():
    dataset_path = download_hf_dataset()

    with tarfile.open(dataset_path) as t:
        extraction_path = f"{ROOT_DIR}/hf_subset"
        print(f"Extracting hugging face dataset into {extraction_path}")
        t.extractall(extraction_path)

        dataset_root_path = pathlib.Path(extraction_path, "UCF101_subset")

    all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.avi"))
        + list(dataset_root_path.glob("val/*/*.avi"))
        + list(dataset_root_path.glob("test/*/*.avi"))
    )

    class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})

    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Unique class labels: {list(label2id.keys())}.")

    train_model(label2id, id2label, dataset_root_path)


def download_hf_dataset() -> str:
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"

    print("Downloading hugging face dataset")
    return hf_hub_download(
        repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"
    )


def train_model(
    label2id: dict[str, int], id2label: dict[int, str], dataset_root_path: str
):
    image_processor, model = load_model(label2id, id2label)

    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]

    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    train_dataset = create_dataset(
        Dataset.TRAIN,
        mean,
        std,
        (height, width),
        num_frames_to_sample,
        dataset_root_path,
        clip_duration,
    )
    print(f"Created training dataset with {train_dataset.num_videos} videos")

    val_dataset = create_dataset(
        Dataset.VALIDATE,
        mean,
        std,
        (height, width),
        num_frames_to_sample,
        dataset_root_path,
        clip_duration,
    )
    print(f"Created validation dataset with {val_dataset.num_videos} videos")

    test_dataset = create_dataset(
        Dataset.EVALUATE,
        mean,
        std,
        (height, width),
        num_frames_to_sample,
        dataset_root_path,
        clip_duration,
    )
    print(f"Created evaluation dataset with {test_dataset.num_videos} videos")

    sample_video = next(iter(train_dataset))
    video_tensor = sample_video["video"]
    display_gif(video_tensor, mean, std)


def create_dataset(
    type: Dataset,
    mean: float,
    std: float,
    resize_to: tuple[int, int],
    num_frames_to_sample: int,
    dataset_root_path: str,
    clip_duration: float,
) -> LabeledVideoDataset:
    base_transform = [
        UniformTemporalSubsample(num_frames_to_sample),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
    ]

    if type == Dataset.TRAIN:  # Apply data augmentation
        base_transform.extend(
            [
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(resize_to),
                RandomHorizontalFlip(p=0.5),
            ]
        )

    dataset_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(base_transform),
            ),
        ]
    )

    sampling_type_name = "random" if type == Dataset.TRAIN else "uniform"
    dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, type.value),
        clip_sampler=pytorchvideo.data.make_clip_sampler(
            sampling_type_name, clip_duration
        ),
        decode_audio=False,
        transform=dataset_transform,
    )

    return dataset


def load_model(
    label2id: dict[str, int], id2label: dict[int, str]
) -> tuple[VideoMAEImageProcessor, VideoMAEForVideoClassification]:
    model_checkpoint = "MCG-NJU/videomae-base"

    image_processor = VideoMAEImageProcessor.from_pretrained(model_checkpoint)

    model = VideoMAEForVideoClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

    return image_processor, model


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


if __name__ == "__main__":
    use_pretrained_hf_model()
