import os
import platform
import subprocess
from typing import Tuple

import cv2


def open_file(file_path):
    """
    Open a file with the default system application, with special handling for WSL2.
    Handles both absolute and relative paths.

    Args:
        file_path: Path to the file to open (absolute or relative)

    Returns:
        True if successful, False otherwise
    """
    # First make sure the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return False

    # Always convert to absolute path to ensure proper handling
    abs_path = os.path.abspath(file_path)

    try:
        system = platform.system()

        # Check if we're running in WSL
        is_wsl = False
        if system == "Linux":
            try:
                with open("/proc/version", "r") as f:
                    if "microsoft" in f.read().lower():
                        is_wsl = True
            except:
                pass

        if system == "Windows":
            os.startfile(abs_path)
        elif system == "Darwin":  # macOS
            subprocess.call(["open", abs_path])
        elif is_wsl:
            _open_file_wsl2(abs_path)
        else:  # Other Linux/Unix systems
            subprocess.call(["xdg-open", abs_path])

        print(f"Opened {abs_path} with system default application")
        return True
    except Exception as e:
        print(f"Failed to open {abs_path}: {e}")
        return False


def _open_file_wsl2(file_path: str):
    """
    Open a file with WSL2.

    Args:
        file_path: Absolute Path to the file to open (absolute or relative)

    Returns:
        True if successful, False otherwise
    """
    # For WSL, we need to convert the path to a Windows path
    try:
        # First check if it's already in the /mnt/ directory
        if file_path.startswith("/mnt/"):
            # Standard WSL mount point conversion
            drive = file_path[5:6]
            path = file_path[6:].replace("/", "\\")
            win_path = f"{drive.upper()}:{path}"
        else:
            # For paths not in /mnt/, we need to:
            # 1. Get the WSL distro path in Windows
            # 2. Append the Linux path (without the leading slash)

            # First, let's get the current working directory's mount point
            # We'll use wslpath to convert between Linux and Windows paths
            wsl_path_process = subprocess.run(
                ["wslpath", "-w", file_path], capture_output=True, text=True
            )

            if wsl_path_process.returncode == 0:
                win_path = wsl_path_process.stdout.strip()

        # Open the file with Windows explorer
        subprocess.call(["explorer.exe", win_path])
        print(f"WSL detected: Opening {win_path} with Windows explorer")
        return True
    except Exception as e:
        print(f"Error converting WSL path: {e}")

    return False


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
