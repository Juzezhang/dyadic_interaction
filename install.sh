#!/bin/bash
# Installation script for Dyadic Interaction Dataset Generator

# Create required directories
mkdir -p /simurgh/group/juze/processed_data/dyadic_interaction/{results,cache,logs,models}

# Create conda environment
echo "Creating conda environment 'dyadic_interaction'..."
conda env create -f environment.yml

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dyadic_interaction

# Download YOLOv8 model for person detection
echo "Downloading YOLOv8 model..."
mkdir -p models
pip install gdown
python -c "import torch; from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "Setting up output directories..."
# Ensure output directories exist with proper permissions
mkdir -p /simurgh/group/juze/processed_data/dyadic_interaction/results
mkdir -p /simurgh/group/juze/processed_data/dyadic_interaction/cache
mkdir -p /simurgh/group/juze/processed_data/dyadic_interaction/logs

echo "Installation complete!"
echo "To activate the environment, run: conda activate dyadic_interaction"
echo ""
echo "To run the pipeline:"
echo "python src/main.py --input_dir /simurgh/group/juze/datasets/YouTube_videos/Talk_video_summary_English_20241226/videos/video_20250304 --output_dir /simurgh/group/juze/processed_data/dyadic_interaction/results" 