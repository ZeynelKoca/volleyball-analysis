from typing import List

import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from torch import Tensor


def display_gif(
    video_tensor: Tensor, mean: float | List[float], std: float | List[float]
):
    """
    Display a video tensor as an animated GIF.
    """
    # Clone to avoid modifying original tensor
    frames = video_tensor.clone().detach().cpu()

    # Ensure the tensor is in [T, C, H, W] format
    if len(frames.shape) != 4:
        raise ValueError(f"Expected 4D tensor [T, C, H, W], got shape {frames.shape}")

    # Check if C is the second dimension (index 1) with size 3
    if frames.shape[1] != 3:
        # If not in expected format, try to automatically detect and permute
        dim_sizes = frames.shape

        # Find the dimension with size 3 (RGB channels)
        channel_dim = next((i for i, size in enumerate(dim_sizes) if size == 3), None)

        if channel_dim is None:
            raise ValueError(
                f"Cannot find channel dimension with size 3 in tensor of shape {frames.shape}"
            )

        # Permute to put channels in the second dimension
        # e.g., if shape is [C, T, H, W], permute to [T, C, H, W]
        if channel_dim == 0:
            frames = frames.permute(1, 0, 2, 3)

        # Add other cases if needed

    # Reshape mean and std for proper broadcasting
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)

    # Apply denormalization
    frames = frames * std_tensor + mean_tensor

    # Clip values to [0, 1] range
    frames = torch.clamp(frames, 0, 1)

    # Convert to numpy and permute to [T, H, W, C] format for plotting
    frames_np = frames.permute(0, 2, 3, 1).numpy()

    # Display animation
    fig, ax = plt.subplots(figsize=(8, 8))

    def animate(i):
        ax.clear()
        ax.imshow(frames_np[i])
        ax.axis("off")
        return [ax]

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_np), interval=100, blit=True
    )

    plt.show()
    return anim
