#!/usr/bin/env python3
"""
Launch script for Sprint Start Analysis application.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_streamlit():
    """Launch Streamlit dashboard."""
    print("Starting Streamlit dashboard...")
    print("Open your browser to: http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app_streamlit.py",
        "--server.port=8501",
        "--server.headless=true"
    ])


def run_api():
    """Launch FastAPI worker."""
    print("Starting FastAPI worker...")
    print("API available at: http://localhost:8000")
    print("Docs available at: http://localhost:8000/docs")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.api.worker_api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


def run_both():
    """Launch both Streamlit and FastAPI."""
    import multiprocessing

    print("Starting both Streamlit dashboard and FastAPI worker...")
    print("Dashboard: http://localhost:8501")
    print("API: http://localhost:8000")

    p1 = multiprocessing.Process(target=run_streamlit)
    p2 = multiprocessing.Process(target=run_api)

    p1.start()
    p2.start()

    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()


def main():
    parser = argparse.ArgumentParser(
        description='Sprint Start Analysis Application Launcher'
    )
    parser.add_argument(
        '--mode',
        choices=['streamlit', 'api', 'both'],
        default='streamlit',
        help='Launch mode: streamlit (dashboard), api (worker), or both'
    )

    args = parser.parse_args()

    if args.mode == 'streamlit':
        run_streamlit()
    elif args.mode == 'api':
        run_api()
    elif args.mode == 'both':
        run_both()


if __name__ == '__main__':
    main()