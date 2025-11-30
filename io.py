import cv2
import json
import numpy as np
from typing import Iterator, Tuple, List, Dict
from pathlib import Path


def extract_frames(video_path: str) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Extract frames from video file.

    Args:
        video_path: Path to video file

    Yields:
        Tuple of (frame_index, frame_array)
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield (frame_idx, frame)
        frame_idx += 1

    cap.release()


def get_video_properties(video_path: str) -> Dict:
    """
    Get video properties.

    Args:
        video_path: Path to video file

    Returns:
        Dict with fps, width, height, frame_count
    """
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count
    }


def write_video_from_frames(frames: List[np.ndarray],
                            out_path: str,
                            fps: float = 30.0):
    """
    Write frames to video file.

    Args:
        frames: List of frame arrays
        out_path: Output video path
        fps: Frames per second
    """
    if not frames:
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


def save_landmarks_json(landmarks_data: List[Dict], out_path: str):
    """
    Save landmarks to JSON file.

    Args:
        landmarks_data: List of detection dicts per frame
        out_path: Output JSON path
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(landmarks_data, f, indent=2)


def load_landmarks_json(json_path: str) -> List[Dict]:
    """
    Load landmarks from JSON file.

    Args:
        json_path: Input JSON path

    Returns:
        List of detection dicts per frame
    """
    with open(json_path, 'r') as f:
        return json.load(f)