from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
import os
import json
from pathlib import Path
import shutil

app = FastAPI(title="Sprint Start Analysis API")

# Storage directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Job storage (in-memory, replace with Redis/DB for production)
jobs: Dict[str, Dict] = {}


class ProcessRequest(BaseModel):
    fps: float = 30.0
    athlete_height_m: Optional[float] = None
    velocity_threshold: float = 0.01
    detector: str = "mediapipe"


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: Optional[str] = None


@app.post("/process")
async def process_video(
        video: UploadFile = File(...),
        fps: float = Form(30.0),
        athlete_height_m: Optional[float] = Form(None),
        velocity_threshold: float = Form(0.01),
        detector: str = Form("mediapipe")
):
    """
    Submit video for processing.

    Returns:
        Job ID for tracking
    """
    job_id = str(uuid.uuid4())

    # Save uploaded video
    video_path = UPLOAD_DIR / f"{job_id}.mp4"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Create job entry
    jobs[job_id] = {
        'status': 'queued',
        'progress': 0.0,
        'video_path': str(video_path),
        'params': {
            'fps': fps,
            'athlete_height_m': athlete_height_m,
            'velocity_threshold': velocity_threshold,
            'detector': detector
        }
    }

    # In production, queue this for background processing
    # For now, we'll process synchronously
    try:
        from ..pose.mediapipe_detector import MediapipeDetector
        from ..processing.metrics import keypoints_to_dataframe, compute_all_metrics
        from ..processing.calibration import Calibrator
        from ..visualization.overlay import create_overlay_video

        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['progress'] = 0.1

        # Detect pose
        detector_obj = MediapipeDetector()
        detections = []
        for frame_idx, det in detector_obj.detect_frames(str(video_path)):
            detections.append(det)
        detector_obj.close()

        jobs[job_id]['progress'] = 0.5

        # Convert to DataFrame
        df = keypoints_to_dataframe(detections, fps)

        # Calibrate if height provided
        calibrator = None
        if athlete_height_m:
            calibrator = Calibrator()
            # Simple auto-calibration
            landmarks_list = [det['landmarks'] for det in detections if det['detected']]
            if landmarks_list:
                calibrator.auto_calibrate(
                    [np.array(lm) for lm in landmarks_list[:10]],
                    athlete_height_m,
                    1920, 1080  # Default, should get from video
                )

        # Compute metrics
        metrics = compute_all_metrics(df, fps, velocity_threshold, calibrator)

        jobs[job_id]['progress'] = 0.8

        # Create overlay video
        result_dir = RESULTS_DIR / job_id
        result_dir.mkdir(exist_ok=True)

        overlay_path = result_dir / "overlay.mp4"
        event_frames = {
            'start': metrics['start_frame'],
            'first_step': metrics['first_step_frame']
        }
        create_overlay_video(str(video_path), detections, str(overlay_path), fps, event_frames)

        # Save results
        results = {
            'metrics': {k: v.tolist() if hasattr(v, 'tolist') else v
                        for k, v in metrics.items()},
            'overlay_video': str(overlay_path)
        }

        with open(result_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['progress'] = 1.0
        jobs[job_id]['result_path'] = str(result_dir)

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = str(e)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """
    Get processing status.

    Args:
        job_id: Job identifier

    Returns:
        Job status
    """
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job['status'],
        progress=job['progress'],
        message=job.get('message')
    )


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Get processing results.

    Args:
        job_id: Job identifier

    Returns:
        Results JSON
    """
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )

    job = jobs[job_id]

    if job['status'] != 'completed':
        return JSONResponse(
            status_code=400,
            content={"error": "Job not completed"}
        )

    result_path = Path(job['result_path']) / "results.json"

    with open(result_path, 'r') as f:
        results = json.load(f)

    return results


@app.get("/download/{job_id}/overlay")
async def download_overlay(job_id: str):
    """
    Download overlay video.

    Args:
        job_id: Job identifier

    Returns:
        Video file
    """
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found"}
        )

    job = jobs[job_id]

    if job['status'] != 'completed':
        return JSONResponse(
            status_code=400,
            content={"error": "Job not completed"}
        )

    overlay_path = Path(job['result_path']) / "overlay.mp4"

    return FileResponse(overlay_path, media_type="video/mp4", filename="overlay.mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)