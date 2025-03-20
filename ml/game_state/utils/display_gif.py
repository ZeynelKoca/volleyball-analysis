import imageio
from matplotlib import animation
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import Tensor


def display_gif(
    video_tensor: Tensor, mean: float, std: float, gif_name: str = "sample.gif"
):
    """
    Prepares and displays a GIF from a video tensor.
    """
    gif_filename = __create_gif(video_tensor, mean, std, gif_name)
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


def __create_gif(
    video_tensor: Tensor, mean: float, std: float, filename: str = "sample.gif"
):
    """
    Prepares a GIF from a video tensor.
    The video tensor is expected to have the following shape:
    (num_frames, num_channels, height, width).
    """
    frames = []
    for video_frame in video_tensor:
        frame_unnormalized = __denormalize_img(
            video_frame.permute(1, 2, 0).numpy(), mean, std
        )
        frames.append(frame_unnormalized)

    kargs = {"duration": 0.25}
    imageio.mimsave(filename, frames, "GIF", **kargs)
    return filename


def __denormalize_img(img: ndarray, mean: float, std: float):
    """
    de-normalizes the image pixels.
    """
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)
