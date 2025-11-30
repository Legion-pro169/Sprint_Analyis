from abc import ABC, abstractmethod
from typing import Dict, List, Iterator, Tuple
import numpy as np


class Detector(ABC):
    """Abstract base class for pose detectors."""

    @abstractmethod
    def detect_frame(self, image: np.ndarray) -> Dict:
        """
        Detect pose in a single frame.

        Args:
            image: BGR image array

        Returns:
            Dict with keys:
                - 'landmarks': List of (x, y, z, visibility) tuples
                - 'detected': bool
        """
        pass

    @abstractmethod
    def detect_frames(self, video_path: str) -> Iterator[Tuple[int, Dict]]:
        """
        Detect pose in all frames of a video.

        Args:
            video_path: Path to video file

        Yields:
            Tuple of (frame_index, detection_dict)
        """
        pass

    @abstractmethod
    def close(self):
        """Release resources."""
        pass