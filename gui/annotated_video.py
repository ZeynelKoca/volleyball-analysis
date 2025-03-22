import gc
import os
from typing import List, Optional

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

from ml.game_state.inference_result import InferenceResult
from ml.game_state.utils.video_utils import get_video_properties


def create_annotated_video(
    video_path: str,
    results: List[InferenceResult],
    output_path: Optional[str] = None,
    chunk_size: int = 30,  # Process this many frames at once
) -> str:
    """
    Create a new video with game state annotations overlaid.
    Process the video in chunks to reduce memory usage.

    Args:
        video_path: Path to the original video
        results: List of InferenceResult objects with predictions
        output_path: Path for the output video (default: input_annotated.mp4)
        chunk_size: Number of frames to process at once

    Returns:
        Path to the annotated video
    """
    # Output path in cwd if none is set
    if output_path is None:
        video_name = os.path.basename(video_path)
        name, ext = os.path.splitext(video_name)
        output_path = f"{name}_annotated{ext}"

    # Get video properties
    total_frames, fps, (height, width), duration = get_video_properties(video_path)
    print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}, {duration:.2f}s")

    # Create a mapping of timestamps to predictions
    prediction_map = {}
    for result in results:
        if result.timestamp is not None and result.top_prediction:
            prediction_map[result.timestamp] = result.top_prediction

    # Get sorted timestamps
    timestamps = sorted(prediction_map.keys())

    # Open video for reading
    input_cap = cv2.VideoCapture(video_path)
    if not input_cap.isOpened():
        raise ValueError(f"Could not open input video: {video_path}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video in chunks to reduce memory usage
    for start_frame in range(0, total_frames, chunk_size):
        end_frame = min(start_frame + chunk_size, total_frames)
        frames_to_process = end_frame - start_frame

        print(f"Processing frames {start_frame}-{end_frame} of {total_frames}")

        # Set frame position
        input_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Process frames in this chunk
        for i in range(frames_to_process):
            ret, frame = input_cap.read()
            if not ret:
                break

            # Calculate the timestamp for this frame
            frame_time = (start_frame + i) / fps

            # Find the most recent prediction for this frame
            current_prediction = None
            current_confidence = 0.0
            for timestamp in timestamps:
                if timestamp <= frame_time:
                    current_prediction = prediction_map[timestamp].label
                    current_confidence = prediction_map[timestamp].confidence
                else:
                    break

            # Add annotation if we have a prediction
            if current_prediction:
                # Create a semi-transparent overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (width - 10, 80), (0, 0, 0), -1)

                # Add text
                cv2.putText(
                    overlay,
                    f"Game State: {current_prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    overlay,
                    f"Confidence: {current_confidence:.2f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                # Apply overlay with transparency
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Write the frame
            output_writer.write(frame)

        # Free some memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Release resources
    input_cap.release()
    output_writer.release()

    print(f"Annotated video created at {output_path}")
    return output_path


def create_kpi_timeline(
    results: List[InferenceResult],
    output_path: Optional[str] = None,
    video_name: Optional[str] = None,
    video_duration: Optional[float] = None,
) -> str:
    """
    Create a timeline visualization with confidence graph.

    Args:
        results: List of InferenceResult objects
        output_path: Path to save the image
        video_name: Name of the video (for title)
        video_duration: Duration of the video in seconds

    Returns:
        Path to the created image
    """
    if not results:
        print("No results to create timeline from")
        return None

    valid_results = [r for r in results if r.timestamp is not None and r.top_prediction]
    if not valid_results:
        print("No valid predictions found in results")
        return None

    if output_path is None:
        output_path = f"{video_name}_timeline.png"

    if video_name is None and valid_results[0].video_name:
        video_name = os.path.basename(valid_results[0].video_name)

    if video_duration is None:
        video_duration = valid_results[-1].timestamp + 2.0  # Add a bit of buffer

    # Set up layout
    width = 1200
    height = 600
    margin = 60
    title_height = 60

    # Set positions for main elements
    confidence_section_y = title_height + margin
    confidence_section_height = 200
    confidence_graph_y = confidence_section_y + confidence_section_height / 2

    states_section_y = confidence_section_y + confidence_section_height + margin * 2
    states_section_height = 120
    timeline_y = states_section_y + states_section_height / 2

    legend_x = width - 300
    legend_y = title_height

    # Create image with white background
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Try to get a nice font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        section_font = ImageFont.truetype("arial.ttf", 18)
        label_font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        try:
            # Try system fonts on different platforms
            system_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "C:/Windows/Fonts/arial.ttf",  # Windows
            ]
            for font_path in system_fonts:
                if os.path.exists(font_path):
                    title_font = ImageFont.truetype(font_path, 24)
                    section_font = ImageFont.truetype(font_path, 18)
                    label_font = ImageFont.truetype(font_path, 14)
                    small_font = ImageFont.truetype(font_path, 12)
                    break
            else:
                raise IOError("No system fonts found")
        except:
            title_font = ImageFont.load_default()
            section_font = title_font
            label_font = title_font
            small_font = title_font

    title = (
        f"Game State Analysis: {video_name}" if video_name else "Game State Analysis"
    )
    draw.text((margin, margin / 2), title, fill=(0, 0, 0), font=title_font)

    # Calculate timeline start and end
    timeline_start = margin
    timeline_end = width - margin
    timeline_width = timeline_end - timeline_start

    unique_states = set(result.top_prediction.label for result in valid_results)

    state_colors = {}
    distinct_colors = [
        (31, 119, 180),  # Blue
        (255, 127, 14),  # Orange
        (44, 160, 44),  # Green
        (214, 39, 40),  # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),  # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
        (188, 189, 34),  # Olive
        (23, 190, 207),  # Teal
    ]
    state_segment_height = 35

    for i, state in enumerate(unique_states):
        color_idx = i % len(distinct_colors)
        state_colors[state] = distinct_colors[color_idx]

    # Draw section labels
    draw.text(
        (margin, confidence_section_y + margin),
        "Confidence (%)",
        fill=(0, 0, 0),
        font=section_font,
    )
    draw.text(
        (margin, states_section_y),
        "Game States",
        fill=(0, 0, 0),
        font=section_font,
    )

    # Draw legend
    draw.text((legend_x, legend_y - 30), "Legend", fill=(0, 0, 0), font=section_font)
    legend_spacing = 30
    current_legend_y = legend_y

    for state, color in state_colors.items():
        draw.rectangle(
            (legend_x, current_legend_y, legend_x + 20, current_legend_y + 20),
            fill=color,
            outline=(0, 0, 0),
        )
        draw.text(
            (legend_x + 30, current_legend_y + 3),
            state,
            fill=(0, 0, 0),
            font=label_font,
        )
        current_legend_y += legend_spacing

    # Draw confidence graph section
    confidence_top = confidence_graph_y
    confidence_bottom = confidence_graph_y + 150

    # Draw graph axes
    draw.line(
        [(timeline_start, confidence_top), (timeline_start, confidence_bottom)],
        fill=(0, 0, 0),
        width=2,
    )
    draw.line(
        [(timeline_start, confidence_bottom), (timeline_end, confidence_bottom)],
        fill=(0, 0, 0),
        width=2,
    )

    # Draw horizontal grid lines and labels for confidence
    grid_color = (200, 200, 200)
    for confidence_level in [0, 25, 50, 75, 100]:
        y = confidence_bottom - (confidence_level / 100) * (
            confidence_bottom - confidence_top
        )

        # Draw grid line
        draw.line([(timeline_start, y), (timeline_end, y)], fill=grid_color, width=1)

        # Add label
        draw.text(
            (timeline_start - 30, y - 7),
            f"{confidence_level}%",
            fill=(0, 0, 0),
            font=small_font,
        )

    # Draw the timeline axis for state timeline
    draw.line(
        [(timeline_start, timeline_y), (timeline_end, timeline_y)],
        fill=(0, 0, 0),
        width=2,
    )

    # Draw time markers on the timeline
    marker_interval = max(1, int(video_duration / 10))  # At most 10 markers
    for t in range(0, int(video_duration) + 1, marker_interval):
        x = timeline_start + (t / video_duration) * timeline_width
        draw.line([(x, timeline_y - 5), (x, timeline_y + 5)], fill=(0, 0, 0), width=1)

        # Only show some time labels to avoid overcrowding
        if t % (marker_interval * 2) == 0 or t == int(video_duration):
            text_bbox = draw.textbbox((0, 0), f"{t}s", font=small_font)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text(
                (x - text_width / 2, timeline_y + state_segment_height),
                f"{t}s",
                fill=(0, 0, 0),
                font=small_font,
            )

    # Draw vertical grid lines for confidence graph
    for t in range(0, int(video_duration) + 1, marker_interval):
        x = timeline_start + (t / video_duration) * timeline_width
        draw.line(
            [(x, confidence_top), (x, confidence_bottom)], fill=grid_color, width=1
        )

    # Plot confidence points and lines
    for state in unique_states:
        state_points = []
        for result in valid_results:
            if result.top_prediction.label == state:
                x = (
                    timeline_start
                    + (result.timestamp / video_duration) * timeline_width
                )
                y = (
                    confidence_bottom
                    - (result.top_prediction.confidence * 100)
                    * (confidence_bottom - confidence_top)
                    / 100
                )
                state_points.append((x, y))

        # Draw lines connecting points of the same state
        if len(state_points) > 1:
            for i in range(len(state_points) - 1):
                draw.line(
                    [state_points[i], state_points[i + 1]],
                    fill=state_colors[state],
                    width=2,
                )

        # Draw points
        for x, y in state_points:
            draw.ellipse(
                (x - 4, y - 4, x + 4, y + 4),
                fill=state_colors[state],
                outline=(0, 0, 0),
            )

    # Process the results to find state segments (for the state timeline)
    state_segments = []
    last_state = None
    last_timestamp = 0

    for i, result in enumerate(valid_results):
        timestamp = result.timestamp
        state = result.top_prediction.label

        if last_state != state:
            if last_state is not None:
                state_segments.append(
                    {
                        "start_time": last_timestamp,
                        "end_time": timestamp,
                        "state": last_state,
                    }
                )
            last_state = state
            last_timestamp = timestamp

    if last_state is not None:
        state_segments.append(
            {
                "start_time": last_timestamp,
                "end_time": video_duration,
                "state": last_state,
            }
        )

    for segment in state_segments:
        start_x = (
            timeline_start + (segment["start_time"] / video_duration) * timeline_width
        )
        end_x = timeline_start + (segment["end_time"] / video_duration) * timeline_width
        color = state_colors.get(segment["state"], (100, 100, 100))

        draw.rectangle(
            [
                (start_x, timeline_y - state_segment_height / 2),
                (end_x, timeline_y + state_segment_height / 2),
            ],
            fill=color,
            outline=(0, 0, 0),
        )

        # Add label if segment is wide enough
        if end_x - start_x > 50:  # Only label segments wide enough to fit text
            text = segment["state"]
            text_bbox = draw.textbbox((0, 0), text, font=label_font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Check if text will fit
            if text_width < (end_x - start_x - 10):
                mid_x = (start_x + end_x) / 2
                text_color = (255, 255, 255)  # White text on colored background
                draw.text(
                    (mid_x - text_width / 2, timeline_y - text_height / 2),
                    text,
                    fill=text_color,
                    font=label_font,
                )

    image.save(output_path)
    print(f"Timeline image created at {output_path}")
    return output_path
