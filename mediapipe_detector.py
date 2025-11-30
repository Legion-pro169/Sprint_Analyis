import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Iterator, Tuple
from .detector_interface import Detector


class MediapipeDetector(Detector):
    """MediaPipe Pose detector implementation."""

    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose detector.

        Args:
            static_image_mode: Whether to treat each image as independent
            model_complexity: 0, 1, or 2 (higher = more accurate, slower)
            min_detection_confidence: Detection confidence threshold
            min_tracking_confidence: Tracking confidence threshold
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_frame(self, image: np.ndarray) -> Dict:
        """
        Detect pose in a single frame.

        Args:
            image: BGR image array

        Returns:
            Dict with 'landmarks' and 'detected' keys
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append((lm.x, lm.y, lm.z, lm.visibility))
            return {
                'landmarks': landmarks,
                'detected': True
            }
        else:
            return {
                'landmarks': [],
                'detected': False
            }

    def detect_frames(self, video_path: str) -> Iterator[Tuple[int, Dict]]:
        """
        Detect pose in all frames of a video.

        Args:
            video_path: Path to video file

        Yields:
            Tuple of (frame_index, detection_dict)
        """
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detection = self.detect_frame(frame)
            yield (frame_idx, detection)
            frame_idx += 1

        cap.release()

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()