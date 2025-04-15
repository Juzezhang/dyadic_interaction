#!/usr/bin/env python
"""
Script to download pre-trained models for the Dyadic Interaction Dataset Generator.
"""

import os
import sys
import argparse
import logging
import yaml
import requests
import urllib.request
from tqdm import tqdm
import zipfile
import torch

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger

# Define model URLs
MODEL_URLS = {
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'smplx': 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip',
    # These are placeholders - actual URLs would need to be provided
    '4d_humans': None,
    'hamer': None,
    'spectre': None,
    'talknet': None
}

def download_file(url, dest_path, desc=None):
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download
        dest_path: Destination path for the file
        desc: Description for the progress bar
    """
    if desc is None:
        desc = os.path.basename(dest_path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def extract_zip(zip_path, extract_dir):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

def download_yolov8(model_dir, model_name='yolov8n'):
    """Download YOLOv8 model."""
    if model_name not in MODEL_URLS or MODEL_URLS[model_name] is None:
        logging.warning(f"No URL available for {model_name}")
        return
    
    dest_path = os.path.join(model_dir, f"{model_name}.pt")
    
    # Skip if already exists
    if os.path.exists(dest_path):
        logging.info(f"{model_name} already exists at {dest_path}")
        return
    
    logging.info(f"Downloading {model_name}...")
    download_file(MODEL_URLS[model_name], dest_path, desc=model_name)
    logging.info(f"Downloaded {model_name} to {dest_path}")

def download_smplx(model_dir):
    """Download SMPLX model."""
    if 'smplx' not in MODEL_URLS or MODEL_URLS['smplx'] is None:
        logging.warning("No URL available for SMPLX")
        return
    
    # Create SMPLX directory
    smplx_dir = os.path.join(model_dir, 'smplx')
    os.makedirs(smplx_dir, exist_ok=True)
    
    # Target file
    zip_path = os.path.join(smplx_dir, 'smplx.zip')
    
    # Skip if model files already exist
    if os.path.exists(os.path.join(smplx_dir, 'SMPLX_NEUTRAL_2020.npz')):
        logging.info(f"SMPLX model already exists in {smplx_dir}")
        return
    
    # Download
    logging.info("Downloading SMPLX model...")
    download_file(MODEL_URLS['smplx'], zip_path, desc='SMPLX')
    
    # Extract
    logging.info("Extracting SMPLX model...")
    extract_zip(zip_path, smplx_dir)
    
    # Clean up
    os.remove(zip_path)
    logging.info(f"SMPLX model extracted to {smplx_dir}")

def download_models(config_path, model_dir=None):
    """
    Download all models specified in the config.
    
    Args:
        config_path: Path to configuration file
        model_dir: Directory to save models to (overrides config)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set model directory
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Check for CPU-only environment
    if not torch.cuda.is_available():
        logging.warning("CUDA not available. Some models may not work or will be slow.")
    
    # Download YOLO model (for person detection)
    yolo_model = config.get('detection', {}).get('model', 'yolov8n.pt')
    yolo_model = yolo_model.split('.')[0]  # Remove file extension
    download_yolov8(model_dir, yolo_model)
    
    # Download SMPLX model
    download_smplx(model_dir)
    
    # Placeholder for other model downloads
    # 4D-Humans, HAMER, SPECTRE, TalkNet would be implemented similarly
    
    logging.info("Model download complete.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model_dir", type=str, default=None,
                       help="Directory to save models to")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Setup logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download models
    download_models(args.config, args.model_dir)

if __name__ == "__main__":
    main() 