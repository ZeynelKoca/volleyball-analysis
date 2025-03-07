from argparse import ArgumentParser, BooleanOptionalAction
from enum import Enum
import imageio
import evaluate
import os
import pathlib
import numpy as np
import tarfile
from matplotlib import animation
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from numpy import ndarray
from torch import Tensor
import torch
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
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


metric = evaluate.load("accuracy")


def main(train_new_model: bool, model_name: str, inference: bool, show_gif: bool):
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

    model_checkpoint = "MCG-NJU/videomae-base"  # Huggingface videomae as base model
    image_processor, model = load_model(model_checkpoint, label2id, id2label)
    train_ds, val_ds, test_ds = get_datasets(image_processor, model, dataset_root_path)

    if train_new_model:
        print("Start training new model...")
        train_results = train_model(
            model, image_processor, model_name, train_ds, val_ds
        )

    if inference:
        print("Start running inference...")
        run_inference(
            model_name=model_name,
            test_dataset=test_ds,
            show_gif=show_gif,
            gif_mean=image_processor.image_mean,
            gif_std=image_processor.image_std,
            dataset_root_path=dataset_root_path,
            id2label=id2label,
        )


def download_hf_dataset() -> str:
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"

    print("Downloading hugging face dataset")
    return hf_hub_download(
        repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"
    )


def train_model(
    model: VideoMAEForVideoClassification,
    image_processor: VideoMAEImageProcessor,
    new_model_name: str,
    training_dataset: LabeledVideoDataset,
    validation_dataset: LabeledVideoDataset,
    num_epochs: int = 20,
    batch_size: int = 8,
):
    args = TrainingArguments(
        output_dir=new_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",  # Often works better than linear
        logging_steps=10,
        load_best_model_at_end=True,
        use_cpu=False,
        fp16=True,  # Exponentially increases training speed on cuda gpu
        dataloader_num_workers=8,  # uses more cpu cores for data loading
        metric_for_best_model="accuracy",
        max_steps=(training_dataset.num_videos // batch_size) * num_epochs,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    return trainer.train()


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


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )

    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def get_datasets(
    image_processor: VideoMAEImageProcessor,
    model: VideoMAEForVideoClassification,
    dataset_root_path: str,
):
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

    return train_dataset, val_dataset, test_dataset


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
    else:
        from torchvision.transforms import Resize

        base_transform.append(Resize(resize_to))

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
    model_checkpoint: str, label2id: dict[str, int], id2label: dict[int, str]
) -> tuple[VideoMAEImageProcessor, VideoMAEForVideoClassification]:
    image_processor = VideoMAEImageProcessor.from_pretrained(model_checkpoint)

    model = VideoMAEForVideoClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    ).to("cuda")

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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train",
        "-T",
        type=bool,
        action=BooleanOptionalAction,
        help="Whether a new model (based on huggingface's videomae base model) should be trained or not",
    )
    parser.add_argument(
        "--name",
        "-N",
        type=str,
        default="videomae-base-finetuned-ucf101-subset/checkpoint-740",
        help="The name of the model. Will be used as the output directory when training a new model, or used to retrieve a pre-trained model from a directory with the same name in the project root",
    )
    parser.add_argument(
        "--inference",
        "-I",
        type=bool,
        action=BooleanOptionalAction,
        help="Whether inference should be done on a test set with either the provided model or the newly trained model",
    )
    parser.add_argument(
        "--demo",
        "-D",
        type=bool,
        action=BooleanOptionalAction,
        help="Whether a demo gif should be shown for each item being inferenced",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.train, args.name, args.inference, args.demo)
