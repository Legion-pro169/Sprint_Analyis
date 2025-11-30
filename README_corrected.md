# Sprint Start Analysis

AI-powered biomechanics analysis system for sprint start movements using MediaPipe pose detection.

## Features

- 🎥 Video-based pose detection using MediaPipe  
- 📊 Comprehensive biomechanics metrics (reaction time, joint angles, velocity, etc.)  
- 🖼️ Annotated video overlay with skeleton and event markers  
- 📈 Interactive visualization dashboard  
- 🔧 Calibration support for real-world measurements  
- 🚀 FastAPI worker for async processing  
- 💻 CLI tool for batch processing  
- ✅ Full test suite  

## Installation

### Requirements
- Python 3.10 or higher  
- ffmpeg (for video processing)

### Setup

```bash
git clone <repository-url>
cd sprint_start_analysis

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Quick Start

### 1. Streamlit Dashboard (Recommended)

```bash
python run_app.py --mode streamlit
```

Then open: **http://localhost:8501**

### 2. Command Line Interface

```bash
python cli/run_analysis.py     --video input_video.mp4     --fps 120     --athlete_height 1.75     --out_dir results/     --save_landmarks
```

### 3. FastAPI Worker

```bash
python run_app.py --mode api
```

API URL: **http://localhost:8000**

---

## Project Structure

```text
sprint_start_analysis/
├── app_streamlit.py
├── run_app.py
├── requirements.txt
├── README.md
├── backend/
│   ├── pose/
│   ├── processing/
│   ├── visualization/
│   ├── utils/
│   └── api/
├── cli/
└── tests/
```

---

## Metrics Computed

### Event Detection
- Movement start  
- First step  

### Temporal Metrics
- Reaction time  
- First step time  

### Kinematics
- Joint angles  
- Angular velocity  
- Trunk lean  
- Horizontal velocity  

---

## Calibration

1. Provide athlete height  
2. Pixel height estimated  
3. Scaling factor computed  
4. Applied to metrics  

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Video Requirements

- Side-view videos  
- Good lighting  
- 720p+ resolution  
- 60 FPS+  
- Full body visible  

---

## License

MIT License

---

Built with ❤️ using MediaPipe, OpenCV, and Streamlit.
