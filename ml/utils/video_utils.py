from typing import Tuple

import cv2


def get_video_properties(video_path: str) -> Tuple[int, float, Tuple[int, int], float]:
    """
    Get properties of a video without loading it entirely into memory.

    Returns:
        Tuple of (total_frames, fps, (height, width), duration_seconds)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    cap.release()

    return total_frames, fps, (height, width), duration


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
