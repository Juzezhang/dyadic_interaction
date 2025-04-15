#!/bin/bash
# Script to run the Dyadic Interaction Dataset Generator

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dyadic_interaction

# Set default values
INPUT_DIR="/simurgh/group/juze/datasets/YouTube_videos/Talk_video_summary_English_20241226/videos/video_20250304"
OUTPUT_DIR="/simurgh/group/juze/processed_data/dyadic_interaction/results"
MAX_VIDEOS=5
DEBUG=false
NO_OVERWRITE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max_videos)
      MAX_VIDEOS="$2"
      shift 2
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --overwrite)
      NO_OVERWRITE=false # Setting this to false means we *should* overwrite
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build command
CMD="python src/main.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --max_videos $MAX_VIDEOS"

if [ "$DEBUG" = true ]; then
  CMD="$CMD --debug"
fi

# Add the --no_overwrite flag *only* if NO_OVERWRITE is true
if [ "$NO_OVERWRITE" = true ]; then
  CMD="$CMD --no_overwrite"
fi

# Print the command
echo "Running command: $CMD"

# Execute the command
eval $CMD 