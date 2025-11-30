#!/usr/bin/env python3
"""
CLI tool for sprint start analysis.
"""
import argparse
import sys
from pathlib import Path
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pose.mediapipe_detector import MediapipeDetector
from backend.processing.metrics import keypoints_to_dataframe, compute_all_metrics
from backend.processing.calibration import Calibrator
from backend.visualization.overlay import create_overlay_video
from backend.utils.io import get_video_properties, save_landmarks_json


def main():
    parser = argparse.ArgumentParser(description='Sprint Start Analysis CLI')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--fps', type=float, default=None, help='Video FPS (auto-detect if not provided)')
    parser.add_argument('--detector', default='mediapipe', choices=['mediapipe'], help='Pose detector')
    parser.add_argument('--out_dir', default='results', help='Output directory')
    parser.add_argument('--athlete_height', type=float, default=None, help='Athlete height in meters')
    parser.add_argument('--velocity_threshold', type=float, default=0.01, help='Movement detection threshold')
    parser.add_argument('--save_landmarks', action='store_true', help='Save raw landmarks JSON')
    parser.add_argument('--no_overlay', action='store_true', help='Skip overlay video creation')

    args = parser.parse_args()

    # Validate input
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Get video properties
    print(f"Loading video: {video_path}")
    props = get_video_properties(str(video_path))
    fps = args.fps if args.fps else props['fps']
    print(f"Video properties: {props['width']}x{props['height']} @ {fps} fps, {props['frame_count']} frames")

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    print(f"Initializing {args.detector} detector...")
    if args.detector == 'mediapipe':
        detector = MediapipeDetector()
    else:
        print(f"Error: Unknown detector: {args.detector}")
        sys.exit(1)

    # Process video
    print("Detecting pose in frames...")
    detections = []
    for frame_idx, det in detector.detect_frames(str(video_path)):
        detections.append(det)
        if (frame_idx + 1) % 30 == 0:
            print(f"  Processed {frame_idx + 1}/{props['frame_count']} frames")

    detector.close()
    print(f"Pose detection complete: {len(detections)} frames")

    # Save landmarks if requested
    if args.save_landmarks:
        landmarks_path = out_dir / "landmarks.json"
        print(f"Saving landmarks to {landmarks_path}")
        save_landmarks_json(detections, str(landmarks_path))

    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = keypoints_to_dataframe(detections, fps)

    # Save DataFrame
    csv_path = out_dir / "keypoints.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved keypoints to {csv_path}")

    # Calibration
    calibrator = None
    if args.athlete_height:
        print(f"Calibrating with athlete height: {args.athlete_height} m")
        calibrator = Calibrator()
        landmarks_list = [np.array(det['landmarks']) for det in detections if det['detected']]
        if landmarks_list:
            success = calibrator.auto_calibrate(
                landmarks_list[:10],
                args.athlete_height,
                props['width'],
                props['height']
            )
            if success:
                print(f"  Calibration successful: {calibrator.scale_factor:.6f} m/px")
            else:
                print("  Calibration failed, using pixel units")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(df, fps, args.velocity_threshold, calibrator)

    # Print summary
    print("\n" + "=" * 60)
    print("SPRINT START ANALYSIS RESULTS")
    print("=" * 60)
    if metrics['start_frame'] is not None:
        print(f"Movement Start: Frame {metrics['start_frame']} ({metrics['reaction_time']:.3f} s)")
    else:
        print("Movement Start: Not detected")

    if metrics['first_step_frame'] is not None:
        print(f"First Step: Frame {metrics['first_step_frame']} ({metrics['first_step_time']:.3f} s after start)")
    else:
        print("First Step: Not detected")

    if metrics['max_velocity'] is not None:
        unit = "m/s" if calibrator else "px/frame"
        print(f"Max Horizontal Velocity: {metrics['max_velocity']:.3f} {unit}")

    print("=" * 60 + "\n")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {}
    for k, v in metrics.items():
        if hasattr(v, 'tolist'):
            metrics_json[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            metrics_json[k] = float(v)
        else:
            metrics_json[k] = v

    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Create overlay video
    if not args.no_overlay:
        print("Creating overlay video...")
        overlay_path = out_dir / "overlay.mp4"
        event_frames = {
            'start': metrics['start_frame'],
            'first_step': metrics['first_step_frame']
        }
        success = create_overlay_video(
            str(video_path),
            detections,
            str(overlay_path),
            fps,
            event_frames
        )
        if success:
            print(f"Saved overlay video to {overlay_path}")
        else:
            print("Failed to create overlay video")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()