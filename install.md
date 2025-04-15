# Dyadic Interaction Dataset Generator

A pipeline system for automatically extracting high-quality multi-modal dyadic interaction data from YouTube videos. The system processes videos to identify segments with exactly two people interacting, extracts their body, hand, and facial motion parameters using state-of-the-art techniques, and aligns this with separated audio streams.

## Features

- **Video Sequence Selection**: Automatically identifies and extracts video segments containing exactly two people interacting
- **Person Tracking**: Maintains consistent identity of each person across frames
- **Holistic Motion Capture**: Extracts body, hand, and facial motion parameters for each person
- **Audio-Visual Speaker Association**: Links speech segments to the corresponding person
- **Audio Separation**: Separates the mixed audio into individual streams for each speaker

## Installation

1. Clone this repository:
```bash
   git clone https://github.com/yourusername/dyadic_interaction.git
   cd dyadic_interaction
   ```

2. Create and activate the conda environment using the install script:
```bash
   chmod +x install.sh
   ./install.sh
   ```

   This will:
   - Create a conda environment named 'dyadic_interaction'
   - Install all required dependencies
   - Download the necessary models
   - Create output directories at `/simurgh/group/juze/processed_data/dyadic_interaction/`

## Usage

### Quick Start

Run the pipeline with default settings using the run script:

```bash
chmod +x run.sh
./run.sh
```

By default, this will:
- Process videos from `/simurgh/group/juze/datasets/YouTube_videos/Talk_video_summary_English_20241226/videos/video_20250304`
- Save results to `/simurgh/group/juze/processed_data/dyadic_interaction/results`
- Process only the first 5 videos (for testing)
- Skip videos that have already been processed

### Custom Run

You can customize the run script with arguments:

```bash
./run.sh --input_dir /path/to/videos --output_dir /path/to/output --max_videos 10 --debug
```

Available options:
- `--input_dir`: Directory containing input videos
- `--output_dir`: Directory to save results
- `--max_videos`: Maximum number of videos to process
- `--debug`: Enable debug logging
- `--overwrite`: Overwrite existing results

### Manual Run

Alternatively, run the pipeline directly using Python:

```bash
conda activate dyadic_interaction
python src/main.py --input_dir /simurgh/group/juze/datasets/YouTube_videos/Talk_video_summary_English_20241226/videos/video_20250304 --output_dir /simurgh/group/juze/processed_data/dyadic_interaction/results --no_overwrite
```

### Configuration

Edit `configs/default.yaml` to adjust pipeline parameters. Key settings include:

```yaml
general:
  output_dir: /simurgh/group/juze/processed_data/dyadic_interaction/results
  cache_dir: /simurgh/group/juze/processed_data/dyadic_interaction/cache
  log_file: /simurgh/group/juze/processed_data/dyadic_interaction/logs/processing.log
  device: cuda:0  # Set to 'cpu' if no CUDA is available

preprocessing:
  fps: 30
  resolution: [720, 1280]
  
detection:
  confidence_threshold: 0.5
  min_segment_length: 30  # Minimum frames for valid two-person segments
```

## Output Format

The pipeline generates dataset entries in the following structure:

```
/simurgh/group/juze/processed_data/dyadic_interaction/results/
  └── video_name/
      ├── metadata.yaml  # Video metadata
      ├── segments/
      │   └── segment_XXX/  # One directory per two-person segment
      │       ├── segment_info.yaml  # Segment metadata
      │       ├── preview/  # Visualization frames
      │       ├── motion/  # Body, hand, face parameters
      │       │   ├── person_1_smplx.pkl
      │       │   └── person_2_smplx.pkl
      │       └── audio/  # Separated audio streams
      │           ├── person_1.wav
      │           └── person_2.wav
```

## Citation

If you use this pipeline in your research, please cite our work:

```bibtex
@misc{dyadic_interaction_dataset,
  author = {Your Name},
  title = {Dyadic Interaction Dataset Generator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/dyadic_interaction}
}
```

## License

MIT License