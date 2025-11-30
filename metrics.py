import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from .signal import smooth_series, compute_velocity

MEDIAPIPE_LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


def keypoints_to_dataframe(detections: List[Dict],
                           fps: float = 30.0) -> pd.DataFrame:
    """
    Convert list of detections to DataFrame.

    Args:
        detections: List of detection dicts from detector
        fps: Video frame rate

    Returns:
        DataFrame with columns for each landmark coordinate
    """
    data = []

    for frame_idx, det in enumerate(detections):
        row = {'frame': frame_idx, 'time': frame_idx / fps}

        if det['detected'] and det['landmarks']:
            landmarks = det['landmarks']
            for i, (x, y, z, vis) in enumerate(landmarks):
                name = MEDIAPIPE_LANDMARK_NAMES[i] if i < len(MEDIAPIPE_LANDMARK_NAMES) else f'point_{i}'
                row[f'{name}_x'] = x
                row[f'{name}_y'] = y
                row[f'{name}_z'] = z
                row[f'{name}_vis'] = vis

        data.append(row)

    return pd.DataFrame(data)


def get_mid_hip_position(df: pd.DataFrame) -> np.ndarray:
    """
    Extract mid-hip position from DataFrame.

    Args:
        df: DataFrame with landmark columns

    Returns:
        Nx2 array of (x, y) positions
    """
    left_hip_x = df['left_hip_x'].fillna(0).values
    left_hip_y = df['left_hip_y'].fillna(0).values
    right_hip_x = df['right_hip_x'].fillna(0).values
    right_hip_y = df['right_hip_y'].fillna(0).values

    mid_x = (left_hip_x + right_hip_x) / 2
    mid_y = (left_hip_y + right_hip_y) / 2

    return np.column_stack([mid_x, mid_y])


def detect_movement_start(df: pd.DataFrame,
                          velocity_threshold: float = 0.01,
                          fps: float = 30.0) -> Optional[int]:
    """
    Detect movement start frame based on hip velocity.

    Args:
        df: DataFrame with landmarks
        velocity_threshold: Threshold for movement (normalized units or m/s)
        fps: Frame rate

    Returns:
        Frame index of movement start, or None
    """
    mid_hip = get_mid_hip_position(df)

    # Compute horizontal velocity
    dt = 1.0 / fps
    x_pos = mid_hip[:, 0]
    x_vel = compute_velocity(x_pos, dt, smooth=True, window_length=5)

    # Find first frame where velocity exceeds threshold
    for i, v in enumerate(x_vel):
        if abs(v) > velocity_threshold:
            return i

    return None


def detect_first_step(df: pd.DataFrame,
                      start_frame: int,
                      fps: float = 30.0,
                      foot_lift_threshold: float = 0.02) -> Optional[int]:
    """
    Detect first step (foot-off) event.

    Args:
        df: DataFrame with landmarks
        start_frame: Movement start frame
        fps: Frame rate
        foot_lift_threshold: Vertical distance threshold

    Returns:
        Frame index of first foot-off, or None
    """
    if start_frame >= len(df):
        return None

    # Get ankle positions after movement start
    left_ankle_y = df['left_ankle_y'].values[start_frame:]
    right_ankle_y = df['right_ankle_y'].values[start_frame:]

    # Find first significant upward movement
    for i in range(1, min(len(left_ankle_y), len(right_ankle_y))):
        left_diff = left_ankle_y[i - 1] - left_ankle_y[i]  # y increases downward
        right_diff = right_ankle_y[i - 1] - right_ankle_y[i]

        if left_diff > foot_lift_threshold or right_diff > foot_lift_threshold:
            return start_frame + i

    return None


def compute_joint_angle(p1: np.ndarray,
                        p2: np.ndarray,
                        p3: np.ndarray) -> float:
    """
    Compute angle at joint p2 formed by points p1-p2-p3.

    Args:
        p1: First point (x, y)
        p2: Joint point (x, y)
        p3: Third point (x, y)

    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return np.degrees(angle)


def compute_joint_angles_over_time(df: pd.DataFrame,
                                   joint_name: str = 'knee') -> np.ndarray:
    """
    Compute joint angles over time.

    Args:
        df: DataFrame with landmarks
        joint_name: 'knee', 'hip', or 'ankle'

    Returns:
        Array of angles in degrees
    """
    angles = []

    for idx in range(len(df)):
        if joint_name == 'knee':
            # Left knee: hip-knee-ankle
            p1 = np.array([df.loc[idx, 'left_hip_x'], df.loc[idx, 'left_hip_y']])
            p2 = np.array([df.loc[idx, 'left_knee_x'], df.loc[idx, 'left_knee_y']])
            p3 = np.array([df.loc[idx, 'left_ankle_x'], df.loc[idx, 'left_ankle_y']])

        elif joint_name == 'hip':
            # Left hip: shoulder-hip-knee
            p1 = np.array([df.loc[idx, 'left_shoulder_x'], df.loc[idx, 'left_shoulder_y']])
            p2 = np.array([df.loc[idx, 'left_hip_x'], df.loc[idx, 'left_hip_y']])
            p3 = np.array([df.loc[idx, 'left_knee_x'], df.loc[idx, 'left_knee_y']])

        elif joint_name == 'ankle':
            # Left ankle: knee-ankle-foot
            p1 = np.array([df.loc[idx, 'left_knee_x'], df.loc[idx, 'left_knee_y']])
            p2 = np.array([df.loc[idx, 'left_ankle_x'], df.loc[idx, 'left_ankle_y']])
            p3 = np.array([df.loc[idx, 'left_foot_index_x'], df.loc[idx, 'left_foot_index_y']])

        else:
            raise ValueError(f"Unknown joint: {joint_name}")

        if not (np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any()):
            angle = compute_joint_angle(p1, p2, p3)
            angles.append(angle)
        else:
            angles.append(np.nan)

    return np.array(angles)


def compute_horizontal_velocity(df: pd.DataFrame,
                                fps: float = 30.0,
                                calibrator=None) -> np.ndarray:
    """
    Compute horizontal COM velocity.

    Args:
        df: DataFrame with landmarks
        fps: Frame rate
        calibrator: Calibrator object for px to m conversion

    Returns:
        Array of horizontal velocities
    """
    mid_hip = get_mid_hip_position(df)
    x_pos = mid_hip[:, 0]

    dt = 1.0 / fps
    x_vel = compute_velocity(x_pos, dt, smooth=True)

    if calibrator:
        x_vel = np.array([calibrator.px_to_m(v) for v in x_vel])

    return x_vel


def compute_trunk_lean_angle(df: pd.DataFrame) -> np.ndarray:
    """
    Compute trunk lean angle from vertical.

    Args:
        df: DataFrame with landmarks

    Returns:
        Array of trunk lean angles in degrees
    """
    angles = []

    for idx in range(len(df)):
        shoulder_x = (df.loc[idx, 'left_shoulder_x'] + df.loc[idx, 'right_shoulder_x']) / 2
        shoulder_y = (df.loc[idx, 'left_shoulder_y'] + df.loc[idx, 'right_shoulder_y']) / 2
        hip_x = (df.loc[idx, 'left_hip_x'] + df.loc[idx, 'right_hip_x']) / 2
        hip_y = (df.loc[idx, 'left_hip_y'] + df.loc[idx, 'right_hip_y']) / 2

        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y

        if not (np.isnan(dx) or np.isnan(dy)):
            angle = np.degrees(np.arctan2(dx, -dy))  # negative dy since y increases downward
            angles.append(angle)
        else:
            angles.append(np.nan)

    return np.array(angles)


def compute_all_metrics(df: pd.DataFrame,
                        fps: float = 30.0,
                        velocity_threshold: float = 0.01,
                        calibrator=None) -> Dict:
    """
    Compute all sprint start metrics.

    Args:
        df: DataFrame with landmarks
        fps: Frame rate
        velocity_threshold: Movement detection threshold
        calibrator: Calibrator object

    Returns:
        Dict of all computed metrics
    """
    # Detect events
    start_frame = detect_movement_start(df, velocity_threshold, fps)
    first_step_frame = None

    if start_frame:
        first_step_frame = detect_first_step(df, start_frame, fps)

    # Compute angles
    knee_angles = compute_joint_angles_over_time(df, 'knee')
    hip_angles = compute_joint_angles_over_time(df, 'hip')
    trunk_lean = compute_trunk_lean_angle(df)

    # Compute velocities
    horizontal_velocity = compute_horizontal_velocity(df, fps, calibrator)

    # Compute times
    reaction_time = start_frame / fps if start_frame else None
    first_step_time = (first_step_frame - start_frame) / fps if (first_step_frame and start_frame) else None

    return {
        'start_frame': start_frame,
        'first_step_frame': first_step_frame,
        'reaction_time': reaction_time,
        'first_step_time': first_step_time,
        'knee_angles': knee_angles,
        'hip_angles': hip_angles,
        'trunk_lean': trunk_lean,
        'horizontal_velocity': horizontal_velocity,
        'max_velocity': np.nanmax(horizontal_velocity) if len(horizontal_velocity) > 0 else None
    }