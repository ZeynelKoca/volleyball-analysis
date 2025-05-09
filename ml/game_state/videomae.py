import glob
import json
import os
import pathlib
import tarfile
from argparse import ArgumentParser, BooleanOptionalAction
from enum import Enum

import evaluate
import numpy as np
import pandas as pd
import pytorchvideo.data
import torch
from huggingface_hub import hf_hub_download
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

from definitions import ROOT_DIR


class Dataset(Enum):
    TRAIN = "train"
    VALIDATE = "val"
    EVALUATE = "test"


metric = evaluate.load("accuracy")


def train(
    model_name: str,
    base_model: str,
    use_hf_dataset: bool,
    dataset_root_path: str | None,
    epochs: int,
    batch_size: int,
):
    if use_hf_dataset:
        dataset_path = download_hf_dataset()

        with tarfile.open(dataset_path) as t:
            extraction_path = f"{ROOT_DIR}/hf_subset"
            print(f"Extracting hugging face dataset into {extraction_path}")
            t.extractall(extraction_path)
            dataset_root_path = pathlib.Path(extraction_path, "UCF101_subset")

    if dataset_root_path is None:
        print("No dataset path was set. Aborting...")
        return

    dataset_root_path = pathlib.Path(dataset_root_path)

    label2id, id2label = get_label_id_dict(dataset_root_path)

    image_processor = VideoMAEImageProcessor.from_pretrained(base_model)
    model = load_model(base_model, label2id, id2label)
    train_ds, val_ds, _ = get_datasets(
        image_processor, model, dataset_root_path, label2id
    )

    print("Start training new model...")
    if torch.cuda.is_available():
        curr_device = torch.cuda.current_device()
        print(f"Using GPU [{torch.cuda.get_device_name(curr_device)}]")

    trainer = create_trainer(
        model=model,
        image_processor=image_processor,
        new_model_name=model_name,
        training_dataset=train_ds,
        validation_dataset=val_ds,
        num_epochs=epochs,
        batch_size=batch_size,
    )

    train_results = trainer.train()

    print(f"Training results: {train_results}")
    report_best_checkpoint(
        checkpoint_dirs=glob.glob(f"ml/output/videomae/{model_name}/checkpoint-*")
    )

    trainer.save_model(f"ml/output/videomae/{model_name}/final")
    print("Finished training and saving model")


def report_best_checkpoint(checkpoint_dirs: list[str]):
    best_accuracy = 0
    best_checkpoint = None

    for checkpoint_dir in checkpoint_dirs:
        state_file = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                # Look at the last entry in the log history for this checkpoint
                if state["log_history"] and "eval_accuracy" in state["log_history"][-1]:
                    accuracy = state["log_history"][-1]["eval_accuracy"]
                    print(f"Checkpoint {checkpoint_dir}: accuracy = {accuracy}")
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_checkpoint = checkpoint_dir

    if best_checkpoint:
        print(f"\nBest checkpoint: {best_checkpoint}")
        print(f"Best accuracy: {best_accuracy}")
    else:
        print("No valid checkpoints found with evaluation metrics.")


def get_label_id_dict(
    dataset_root_path: pathlib.Path,
) -> tuple[dict[str, int], dict[int, str]]:
    class_labels = [
        name
        for name in os.listdir(dataset_root_path)
        if pathlib.Path.is_dir(pathlib.Path(dataset_root_path, name))
    ]

    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Found unique class labels: {list(label2id.keys())}.")

    return label2id, id2label


def download_hf_dataset() -> str:
    hf_dataset_identifier = "sayakpaul/ucf101-subset"
    filename = "UCF101_subset.tar.gz"

    print("Downloading hugging face dataset")
    return hf_hub_download(
        repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"
    )


def create_trainer(
    model: VideoMAEForVideoClassification,
    image_processor: VideoMAEImageProcessor,
    new_model_name: str,
    training_dataset: LabeledVideoDataset,
    validation_dataset: LabeledVideoDataset,
    num_epochs: int = 20,
    batch_size: int = 8,
):
    args = TrainingArguments(
        output_dir=f"ml/output/videomae/{new_model_name}",
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
        use_cpu=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),  # Exponentially increases training speed on cuda gpu
        metric_for_best_model="accuracy",
        max_steps=(training_dataset.num_videos // batch_size) * num_epochs,
    )

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        processing_class=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[early_stopping_callback],
    )

    return trainer


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
    label2id: dict[str, int],
    random_seed: int = 42,  # For dataset reproducability
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

    # Collect all video files and their labels
    video_files = []
    labels = []
    file_extensions = ["mp4", "mov", "avi", "mkv"]

    for class_dir in dataset_root_path.glob("*"):
        if class_dir.is_dir():
            class_name = class_dir.name

            # Collect files with any of the specified extensions
            for ext in file_extensions:
                for video_file in class_dir.glob(f"*.{ext}"):
                    video_files.append(str(video_file))
                    labels.append(class_name)

    if not video_files:
        raise ValueError(
            f"No video files found with extensions {file_extensions} in {dataset_root_path}"
        )

    df = pd.DataFrame({"path": video_files, "label": labels})
    print(f"Found {len(df)} videos across {df['label'].nunique()} classes")
    # Add numerical label column
    df["label_id"] = df["label"].map(label2id)

    # First split: train and temp (val+test combined)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=random_seed
    )
    # Second split: val and test from temp
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=random_seed
    )

    print(
        f"Split dataset: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test videos"
    )

    train_dataset = create_dataset(
        train_df,
        Dataset.TRAIN,
        mean,
        std,
        (height, width),
        num_frames_to_sample,
        clip_duration,
    )
    print(f"Created training dataset with {train_dataset.num_videos} videos")

    val_dataset = create_dataset(
        val_df,
        Dataset.VALIDATE,
        mean,
        std,
        (height, width),
        num_frames_to_sample,
        clip_duration,
    )
    print(f"Created validation dataset with {val_dataset.num_videos} videos")

    test_dataset = create_dataset(
        test_df,
        Dataset.EVALUATE,
        mean,
        std,
        (height, width),
        num_frames_to_sample,
        clip_duration,
    )
    print(f"Created evaluation dataset with {test_dataset.num_videos} videos")

    return train_dataset, val_dataset, test_dataset


def create_dataset(
    df: pd.DataFrame,
    dataset_type: Dataset,
    mean: float,
    std: float,
    resize_to: tuple[int, int],
    num_frames_to_sample: int,
    clip_duration: float,
) -> LabeledVideoDataset:
    base_transform = [
        UniformTemporalSubsample(num_frames_to_sample),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
    ]

    if dataset_type == Dataset.TRAIN:  # Apply data augmentation
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

    if dataset_type == Dataset.EVALUATE:
        # Use constant sampler with 1 clip per video
        clip_sampler = pytorchvideo.data.make_clip_sampler(
            "constant_clips_per_video", clip_duration, 1
        )
    else:
        sampling_type_name = "random" if dataset_type == Dataset.TRAIN else "uniform"
        clip_sampler = pytorchvideo.data.make_clip_sampler(
            sampling_type_name, clip_duration
        )

    video_data = [(row["path"], {"label": row["label_id"]}) for _, row in df.iterrows()]

    dataset = LabeledVideoDataset(
        video_data,
        clip_sampler=clip_sampler,
        decode_audio=False,
        transform=dataset_transform,
    )

    return dataset


def load_model(
    model_checkpoint: str, label2id: dict[str, int], id2label: dict[int, str]
) -> tuple[VideoMAEImageProcessor, VideoMAEForVideoClassification]:
    model = VideoMAEForVideoClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )

    if torch.cuda.is_available():
        model.to("cuda")

    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--name",
        "-N",
        type=str,
        default="videomae-base-finetuned-ucf101-subset",
        help="The name of the new model. Will be used as the parent output directory when training a new model",
    )
    parser.add_argument(
        "--base-model",
        "-B",
        type=str,
        default="MCG-NJU/videomae-base",
        help="The (huggingface) base model which will be used to train a new model using custom datasets.",
    )
    parser.add_argument(
        "--hf-ucf101-subset",
        "-H",
        type=bool,
        action=BooleanOptionalAction,
        help="Whether the UCF101 dataset should be used (and downloaded) from huggingface. If not set, make sure to set --dataset-path",
    )
    parser.add_argument(
        "--dataset-path",
        "-D",
        type=str,
        help="The directory (residing in project root) that contains the custom dataset. Directory should consist of sub-folders with label names, where video data resides. This argument will be ignored when using --hf-ucf101-subset",
    )
    parser.add_argument(
        "--epochs",
        "-E",
        type=int,
        default=20,
        help="Maximum epoch count",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        args.name,
        args.base_model,
        args.hf_ucf101_subset,
        args.dataset_path,
        args.epochs,
        args.batch_size,
    )
