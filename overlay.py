import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import mediapipe as mp

# MediaPipe pose connections
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10)
]


def draw_landmarks_on_frame(frame: np.ndarray,
                            landmarks: List[Tuple],
                            connections: List[Tuple] = None) -> np.ndarray:
    """
    Draw pose landmarks and connections on frame.

    Args:
        frame: Input frame
        landmarks: List of (x, y, z, visibility) tuples (normalized 0-1)
        connections: List of (idx1, idx2) tuples for skeleton

    Returns:
        Frame with overlay
    """
    if not landmarks:
        return frame

    height, width = frame.shape[:2]
    overlay = frame.copy()

    # Convert normalized to pixel coordinates
    points = []
    for x, y, z, vis in landmarks:
        px = int(x * width)
        py = int(y * height)
        points.append((px, py, vis))

    # Draw connections
    if connections is None:
        connections = POSE_CONNECTIONS

    for idx1, idx2 in connections:
        if idx1 < len(points) and idx2 < len(points):
            p1, p2 = points[idx1], points[idx2]
            if p1[2] > 0.5 and p2[2] > 0.5:  # visibility threshold
                cv2.line(overlay, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), 2)

    # Draw landmarks
    for px, py, vis in points:
        if vis > 0.5:
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)

    return overlay


def draw_event_marker(frame: np.ndarray,
                      text: str,
                      position: Tuple[int, int] = (50, 50),
                      color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    Draw event marker text on frame.

    Args:
        frame: Input frame
        text: Marker text
        position: (x, y) position
        color: BGR color

    Returns:
        Frame with text
    """
    overlay = frame.copy()
    cv2.putText(overlay, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)
    return overlay


def create_overlay_video(video_path: str,
                         detections: List[Dict],
                         output_path: str,
                         fps: float = 30.0,
                         event_frames: Optional[Dict[str, int]] = None) -> bool:
    """
    Create overlay video with pose detection and events.

    Args:
        video_path: Input video path
        detections: List of detection dicts
        output_path: Output video path
        fps: Frame rate
        event_frames: Dict of event_name -> frame_idx

    Returns:
        Success status
    """
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw pose if detected
        if frame_idx < len(detections):
            det = detections[frame_idx]
            if det['detected'] and det['landmarks']:
                frame = draw_landmarks_on_frame(frame, det['landmarks'])

        # Draw event markers
        if event_frames:
            if frame_idx == event_frames.get('start', -1):
                frame = draw_event_marker(frame, "MOVEMENT START", (50, 50), (0, 255, 0))

            if frame_idx == event_frames.get('first_step', -1):
                frame = draw_event_marker(frame, "FIRST STEP", (50, 100), (255, 255, 0))

        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    return True


def create_overlay_frames(video_path: str,
                          detections: List[Dict],
                          event_frames: Optional[Dict[str, int]] = None) -> List[np.ndarray]:
    """
    Create list of overlay frames without writing to disk.

    Args:
        video_path: Input video path
        detections: List of detection dicts
        event_frames: Dict of event_name -> frame_idx

    Returns:
        List of overlay frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw pose if detected
        if frame_idx < len(detections):
            det = detections[frame_idx]
            if det['detected'] and det['landmarks']:
                frame = draw_landmarks_on_frame(frame, det['landmarks'])

        # Draw event markers
        if event_frames:
            if frame_idx == event_frames.get('start', -1):
                frame = draw_event_marker(frame, "MOVEMENT START", (50, 50), (0, 255, 0))

            if frame_idx == event_frames.get('first_step', -1):
                frame = draw_event_marker(frame, "FIRST STEP", (50, 100), (255, 255, 0))

        frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames