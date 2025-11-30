import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import cv2
from pathlib import Path
import json

from backend.pose.mediapipe_detector import MediapipeDetector
from backend.processing.metrics import (
    keypoints_to_dataframe,
    compute_all_metrics,
    MEDIAPIPE_LANDMARK_NAMES
)
from backend.processing.calibration import Calibrator
from backend.visualization.overlay import create_overlay_video
from backend.utils.io import get_video_properties

st.set_page_config(page_title="Sprint Start Analysis", layout="wide")

st.title("🏃 Sprint Start Biomechanics Analysis")
st.markdown("Upload a sprint start video for AI-powered pose detection and biomechanics analysis")

# Sidebar configuration
st.sidebar.header("Configuration")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'detections' not in st.session_state:
    st.session_state.detections = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# Video upload
uploaded_file = st.file_uploader("Upload Sprint Start Video", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
        st.session_state.video_path = video_path

    # Get video properties
    props = get_video_properties(video_path)

    col1, col2 = st.columns(2)
    with col1:
        st.video(video_path)
    with col2:
        st.subheader("Video Properties")
        st.write(f"**Resolution:** {props['width']}x{props['height']}")
        st.write(f"**Frame Rate:** {props['fps']:.2f} fps")
        st.write(f"**Total Frames:** {props['frame_count']}")
        st.write(f"**Duration:** {props['frame_count'] / props['fps']:.2f} seconds")

    # Configuration
    st.sidebar.subheader("Analysis Parameters")

    fps = st.sidebar.number_input(
        "Frame Rate (fps)",
        min_value=1.0,
        max_value=240.0,
        value=props['fps'],
        help="Override detected FPS if needed"
    )

    athlete_height = st.sidebar.number_input(
        "Athlete Height (m)",
        min_value=0.0,
        max_value=3.0,
        value=1.75,
        step=0.01,
        help="Athlete height for calibration (optional)"
    )

    use_calibration = st.sidebar.checkbox("Use Calibration", value=True)

    velocity_threshold = st.sidebar.slider(
        "Movement Detection Threshold",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Velocity threshold for detecting movement start"
    )

    # Process button
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Processing video... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Detect pose
            status_text.text("Step 1/5: Detecting pose in frames...")
            detector = MediapipeDetector()
            detections = []

            frame_count = props['frame_count']
            for frame_idx, det in detector.detect_frames(video_path):
                detections.append(det)
                if frame_idx % 10 == 0:
                    progress_bar.progress(min(0.3 * frame_idx / frame_count, 0.3))

            detector.close()
            st.session_state.detections = detections
            progress_bar.progress(0.3)

            # Convert to DataFrame
            status_text.text("Step 2/5: Converting to time-series data...")
            df = keypoints_to_dataframe(detections, fps)
            st.session_state.df = df
            progress_bar.progress(0.5)

            # Calibration
            calibrator = None
            if use_calibration and athlete_height > 0:
                status_text.text("Step 3/5: Calibrating measurements...")
                calibrator = Calibrator()
                landmarks_list = [np.array(det['landmarks']) for det in detections if det['detected']]
                if landmarks_list:
                    calibrator.auto_calibrate(
                        landmarks_list[:10],
                        athlete_height,
                        props['width'],
                        props['height']
                    )
            progress_bar.progress(0.6)

            # Compute metrics
            status_text.text("Step 4/5: Computing biomechanics metrics...")
            metrics = compute_all_metrics(df, fps, velocity_threshold, calibrator)
            st.session_state.results = metrics
            progress_bar.progress(0.8)

            # Create overlay video
            status_text.text("Step 5/5: Creating overlay video...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='_overlay.mp4') as tmp_overlay:
                overlay_path = tmp_overlay.name
                event_frames = {
                    'start': metrics['start_frame'],
                    'first_step': metrics['first_step_frame']
                }
                create_overlay_video(video_path, detections, overlay_path, fps, event_frames)
                st.session_state.overlay_path = overlay_path

            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            st.success("Analysis completed successfully!")

# Display results
if st.session_state.results:
    st.header("📊 Analysis Results")

    metrics = st.session_state.results
    df = st.session_state.df

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Reaction Time",
            f"{metrics['reaction_time']:.3f} s" if metrics['reaction_time'] else "N/A",
            help="Time from start to first movement"
        )

    with col2:
        st.metric(
            "First Step Time",
            f"{metrics['first_step_time']:.3f} s" if metrics['first_step_time'] else "N/A",
            help="Time from movement start to first foot-off"
        )

    with col3:
        unit = "m/s" if use_calibration else "px/frame"
        st.metric(
            "Max Velocity",
            f"{metrics['max_velocity']:.2f} {unit}" if metrics['max_velocity'] else "N/A",
            help="Maximum horizontal velocity"
        )

    with col4:
        detected_frames = sum(1 for d in st.session_state.detections if d['detected'])
        st.metric(
            "Detection Rate",
            f"{100 * detected_frames / len(st.session_state.detections):.1f}%",
            help="Percentage of frames with successful pose detection"
        )

    # Overlay video
    st.subheader("Annotated Video")
    if hasattr(st.session_state, 'overlay_path'):
        st.video(st.session_state.overlay_path)

    # Charts
    st.subheader("Biomechanics Plots")

    tab1, tab2, tab3 = st.tabs(["Joint Angles", "Velocity", "Trunk Lean"])

    with tab1:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        time = df['time'].values

        # Knee angle
        axes[0].plot(time, metrics['knee_angles'], 'b-', linewidth=2)
        axes[0].set_ylabel('Knee Angle (°)', fontsize=12)
        axes[0].set_title('Left Knee Angle Over Time', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        if metrics['start_frame']:
            axes[0].axvline(time[metrics['start_frame']], color='g', linestyle='--', label='Start')
        if metrics['first_step_frame']:
            axes[0].axvline(time[metrics['first_step_frame']], color='r', linestyle='--', label='First Step')
        axes[0].legend()

        # Hip angle
        axes[1].plot(time, metrics['hip_angles'], 'r-', linewidth=2)
        axes[1].set_ylabel('Hip Angle (°)', fontsize=12)
        axes[1].set_title('Left Hip Angle Over Time', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        if metrics['start_frame']:
            axes[1].axvline(time[metrics['start_frame']], color='g', linestyle='--')
        if metrics['first_step_frame']:
            axes[1].axvline(time[metrics['first_step_frame']], color='r', linestyle='--')

        # Trunk lean
        axes[2].plot(time, metrics['trunk_lean'], 'purple', linewidth=2)
        axes[2].set_ylabel('Trunk Lean (°)', fontsize=12)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_title('Trunk Lean Angle Over Time', fontsize=14)
        axes[2].grid(True, alpha=0.3)
        if metrics['start_frame']:
            axes[2].axvline(time[metrics['start_frame']], color='g', linestyle='--')
        if metrics['first_step_frame']:
            axes[2].axvline(time[metrics['first_step_frame']], color='r', linestyle='--')

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 6))

        unit = "m/s" if use_calibration else "px/frame"
        ax.plot(time, metrics['horizontal_velocity'], 'b-', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(f'Horizontal Velocity ({unit})', fontsize=12)
        ax.set_title('Horizontal COM Velocity Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)

        if metrics['start_frame']:
            ax.axvline(time[metrics['start_frame']], color='g', linestyle='--', label='Start', linewidth=2)
        if metrics['first_step_frame']:
            ax.axvline(time[metrics['first_step_frame']], color='r', linestyle='--', label='First Step', linewidth=2)

        ax.legend(fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(time, metrics['trunk_lean'], 'purple', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Trunk Lean Angle (°)', fontsize=12)
        ax.set_title('Trunk Lean Angle Over Time', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=1)

        if metrics['start_frame']:
            ax.axvline(time[metrics['start_frame']], color='g', linestyle='--', label='Start', linewidth=2)
        if metrics['first_step_frame']:
            ax.axvline(time[metrics['first_step_frame']], color='r', linestyle='--', label='First Step', linewidth=2)

        ax.legend(fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)

    # Data table
    st.subheader("Keypoint Data")

    # Show sample of data
    display_cols = ['frame', 'time'] + [col for col in df.columns if '_x' in col or '_y' in col][:10]
    st.dataframe(df[display_cols].head(50), use_container_width=True)

    # Download options
    st.subheader("📥 Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Keypoints CSV",
            data=csv_data,
            file_name="keypoints.csv",
            mime="text/csv"
        )

    with col2:
        # Convert metrics to JSON-serializable format
        metrics_json = {}
        for k, v in metrics.items():
            if hasattr(v, 'tolist'):
                metrics_json[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                metrics_json[k] = float(v)
            else:
                metrics_json[k] = v

        json_data = json.dumps(metrics_json, indent=2)
        st.download_button(
            label="Download Metrics JSON",
            data=json_data,
            file_name="metrics.json",
            mime="application/json"
        )

    with col3:
        if hasattr(st.session_state, 'overlay_path'):
            with open(st.session_state.overlay_path, 'rb') as f:
                st.download_button(
                    label="Download Overlay Video",
                    data=f,
                    file_name="overlay.mp4",
                    mime="video/mp4"
                )

else:
    st.info("👆 Upload a video and click 'Run Analysis' to begin")

# Footer
st.markdown("---")
st.markdown("""
**Sprint Start Analysis v1.0** | Built with MediaPipe, OpenCV, and Streamlit  
For best results, ensure good lighting and a clear side view of the athlete.
""")