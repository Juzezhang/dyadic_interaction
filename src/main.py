#!/usr/bin/env python
"""
Main entry point for the Dyadic Interaction Dataset Generator.

This script orchestrates the entire pipeline from video preprocessing
to motion extraction, audio processing, and data integration.
"""

import os
import sys
import argparse
import yaml
import logging
from typing import Dict, List, Any, Optional
import torch
import shutil
from pathlib import Path
import datetime
import cv2
import pickle
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from src.utils.logger import setup_logger, log_config
from src.preprocessing.preprocessor import VideoPreprocessor
from src.detection.person_detector import PersonDetector
from src.tracking.person_tracker import PersonTracker
from src.pose_estimation.pose_estimator import PoseEstimator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dyadic Interaction Dataset Generator")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (default: from config)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Maximum number of videos to process")
    parser.add_argument("--no_overwrite", action="store_true",
                       help="Don't overwrite existing results")
    parser.add_argument("--skip_to_step", type=str, default=None,
                       help="Skip to a specific step (e.g., 'tracking', 'pose')")
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Update configuration with command line arguments."""
    # Update general settings
    if args.debug:
        config['general']['debug'] = True
        config['general']['log_level'] = 'debug'
    
    if args.output_dir:
        config['general']['output_dir'] = args.output_dir
    
    if args.device:
        config['general']['device'] = args.device
    
    # Ensure output paths are separate from input paths
    input_dir = args.input_dir
    output_dir = config['general']['output_dir']
    
    # Check if output_dir is inside input_dir
    if os.path.commonpath([os.path.abspath(input_dir)]) == os.path.commonpath([os.path.abspath(input_dir), os.path.abspath(output_dir)]):
        # If output is inside input, add a timestamp to make it unique
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_output = output_dir
        output_dir = f"{output_dir}_{timestamp}"
        config['general']['output_dir'] = output_dir
        print(f"Warning: Output directory was inside input directory. Changed from {original_output} to {output_dir}")
    
    # Add timestamp to log file if it exists in config
    if 'log_file' in config['general'] and config['general']['log_file']:
        log_path = Path(config['general']['log_file'])
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config['general']['log_file'] = str(log_path.parent / f"{log_path.stem}_{timestamp}{log_path.suffix}")
    
    # Pass global cache_dir to preprocessing config if not set there
    if 'cache_dir' not in config['preprocessing']:
        config['preprocessing']['cache_dir'] = config['general'].get('cache_dir', 'data/cache')
        
    # Pass global device to detection config if not set there
    if 'device' not in config['detection']:
        config['detection']['device'] = config['general'].get('device', 'cuda:0')

    return config

def setup_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories for the pipeline."""
    os.makedirs(config['general']['output_dir'], exist_ok=True)
    os.makedirs(config['general']['cache_dir'], exist_ok=True)
    
    # Create logs directory if log_file is specified
    if 'log_file' in config['general'] and config['general']['log_file']:
        log_dir = os.path.dirname(config['general']['log_file'])
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

def get_video_files(input_dir: str) -> List[str]:
    """Get list of video files in the input directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return sorted(video_files) # Sort for consistent processing order

def get_output_path_for_video(video_path: str, output_dir: str) -> str:
    """Generate a unique output path for a video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, f"{video_name}")

def save_visualizations(frames: List[np.ndarray], detections: List[List[Dict[str, Any]]], detector: PersonDetector, output_path: str, logger: logging.Logger):
    """Save frames with detection visualizations."""
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Saving detection visualizations to {output_path}")
    for i, (frame, frame_detections) in enumerate(zip(frames, detections)):
        if frame_detections: # Only save frames with detections
            viz_frame = detector.visualize_detections(frame, frame_detections)
            cv2.imwrite(os.path.join(output_path, f"frame_{i:06d}.jpg"), viz_frame)

def save_tracking_visualizations(frames: List[np.ndarray], tracker: PersonTracker, tracked_segment: Dict[str, Any], output_path: str, logger: logging.Logger):
    """Save frames with tracking visualizations."""
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Saving tracking visualizations to {output_path}")
    
    # Create visualization frames
    viz_frames = tracker.create_track_visualization(tracked_segment, frames)
    
    # Save visualization frames
    for i, viz_frame in enumerate(viz_frames):
        cv2.imwrite(os.path.join(output_path, f"frame_{i:06d}.jpg"), viz_frame)

def save_pose_visualizations(frames: List[np.ndarray], pose_estimator: PoseEstimator, segment_with_pose: Dict[str, Any], output_path: str, logger: logging.Logger):
    """Save frames with pose estimation visualizations."""
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Saving pose visualizations to {output_path}")
    
    # Create visualization frames
    viz_frames = pose_estimator.create_pose_visualization(segment_with_pose, frames)
    
    # Save visualization frames
    for i, viz_frame in enumerate(viz_frames):
        cv2.imwrite(os.path.join(output_path, f"frame_{i:06d}.jpg"), viz_frame)

def process_video(video_path: str, config: Dict[str, Any], logger: logging.Logger, no_overwrite: bool = False, skip_to_step: Optional[str] = None) -> None:
    """Process a single video through the pipeline."""
    # Generate output directory for this video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = get_output_path_for_video(video_path, config['general']['output_dir'])
    
    # --- Check if already processed --- 
    completion_flag = os.path.join(video_output_dir, 'processing_complete.flag')
    if no_overwrite and os.path.exists(completion_flag):
        logger.info(f"Skipping already completed video: {video_path}")
        return
    elif no_overwrite and os.path.exists(video_output_dir):
         logger.info(f"Output directory exists but not marked complete, potentially resuming/overwriting: {video_output_dir}")
         # Decide if you want to remove existing directory or attempt resume
         # For now, we proceed and overwrite intermediate steps
    elif os.path.exists(video_output_dir):
        logger.warning(f"Overwriting existing output directory: {video_output_dir}")
        # shutil.rmtree(video_output_dir) # Uncomment to force clean overwrite

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output directory: {video_output_dir}")
    
    # --- Create video-specific directories --- 
    os.makedirs(video_output_dir, exist_ok=True)
    detections_viz_dir = os.path.join(video_output_dir, 'visualization', 'detections')
    tracking_viz_dir = os.path.join(video_output_dir, 'visualization', 'tracking')
    pose_viz_dir = os.path.join(video_output_dir, 'visualization', 'poses')
    segments_dir = os.path.join(video_output_dir, 'segments')
    os.makedirs(detections_viz_dir, exist_ok=True)
    os.makedirs(tracking_viz_dir, exist_ok=True)
    os.makedirs(pose_viz_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)
    
    # --- Initialize components --- 
    preprocessor = VideoPreprocessor(config['preprocessing'])
    detector = PersonDetector(config['detection'])
    tracker = PersonTracker(config['tracking'])
    pose_estimator = PoseEstimator(config['motion']['body'])
    
    # --- Step 1: Preprocess video --- 
    frames = None
    scene_changes = None
    metadata = None
    
    if skip_to_step in ['tracking', 'pose', 'audio', 'integration']:
        # Try to load frames from cache if skipping to later steps
        cache_dir = config['preprocessing'].get('cache_dir', config['general'].get('cache_dir', 'data/cache'))
        cache_file = os.path.join(cache_dir, f"{video_name}_frames.pkl")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading preprocessed frames from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    frames = cached_data.get('frames')
                    scene_changes = cached_data.get('scene_changes', [])
                    metadata = cached_data.get('metadata', {})
            except Exception as e:
                logger.warning(f"Failed to load frames from cache: {e}")
        
        # Load metadata if exists
        metadata_file = os.path.join(video_output_dir, 'metadata.yaml')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata_yaml = yaml.safe_load(f)
                    if not scene_changes:
                        scene_changes = metadata_yaml.get('scene_changes', [])
                    if not metadata:
                        metadata = metadata_yaml.get('original_metadata', {})
            except Exception as e:
                logger.warning(f"Failed to load metadata from file: {e}")
                
    if frames is None or scene_changes is None:
        try:
            preprocessing_results = preprocessor.process_video(video_path)
            frames = preprocessing_results.get('frames')
            scene_changes = preprocessing_results.get('scene_changes')
            metadata = preprocessing_results.get('metadata')
            
            if not frames:
                logger.error(f"Preprocessing failed or returned no frames for {video_path}")
                return # Cannot proceed without frames
                
            logger.info(f"Preprocessed video: {len(frames)} frames, {len(scene_changes)} scene changes")
            
            # Save metadata
            with open(os.path.join(video_output_dir, 'metadata.yaml'), 'w') as f:
                yaml.dump({
                    'original_video_path': video_path,
                    'frame_count': len(frames),
                    'scene_changes': scene_changes,
                    'original_metadata': metadata,
                    'processing_config': config,
                }, f)
                
        except Exception as e:
            logger.error(f"Error during preprocessing for {video_path}: {e}", exc_info=True)
            return # Stop processing this video

    # --- Step 2: Detect persons in frames --- 
    detections = None
    
    if skip_to_step in ['tracking', 'pose', 'audio', 'integration']:
        # Try to load detections if skipping to later steps
        detections_file = os.path.join(video_output_dir, 'raw_detections.pkl')
        if os.path.exists(detections_file):
            logger.info(f"Loading detections from file: {detections_file}")
            try:
                with open(detections_file, 'rb') as f:
                    detections = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load detections from file: {e}")
    
    if detections is None:
        try:
            detections = detector.detect_persons(frames)
            logger.info(f"Detected persons in {sum(1 for d in detections if d)} frames")
            
            # Save raw detections
            with open(os.path.join(video_output_dir, 'raw_detections.pkl'), 'wb') as f:
                pickle.dump(detections, f)
            
            # Save detection visualizations if enabled
            if config.get('detection', {}).get('visualization', False):
                save_visualizations(frames, detections, detector, detections_viz_dir, logger)
                
        except Exception as e:
            logger.error(f"Error during person detection for {video_path}: {e}", exc_info=True)
            return # Stop processing this video

    # --- Step 3: Filter for two-person segments --- 
    segments = None
    
    if skip_to_step in ['tracking', 'pose', 'audio', 'integration']:
        # Try to load segments if skipping to later steps
        segments_file = os.path.join(video_output_dir, 'segments_summary.yaml')
        if os.path.exists(segments_file):
            logger.info(f"Loading segments from file: {segments_file}")
            try:
                with open(segments_file, 'r') as f:
                    segments_info = yaml.safe_load(f)
                    
                segments = []
                for segment_info in segments_info:
                    segment_id = segment_info.get('segment_id')
                    segment_dir = os.path.join(segments_dir, f"segment_{segment_id:03d}")
                    
                    # Load segment detections
                    detections_file = os.path.join(segment_dir, 'detections.pkl')
                    if os.path.exists(detections_file):
                        with open(detections_file, 'rb') as f:
                            segment_detections = pickle.load(f)
                            
                        segment_data = {
                            'segment_id': segment_id,
                            'start_idx': segment_info.get('start_frame_idx'),
                            'end_idx': segment_info.get('end_frame_idx'),
                            'length': segment_info.get('length_frames'),
                            'detections': segment_detections
                        }
                        segments.append(segment_data)
            except Exception as e:
                logger.warning(f"Failed to load segments from file: {e}")
    
    if segments is None:
        try:
            segments = detector.filter_two_person_segments(
                frames=frames,
                detections=detections,
                scene_changes=scene_changes,
                min_segment_length=config['detection']['min_segment_length']
            )
            logger.info(f"Found {len(segments)} two-person segments")
            
            # Save segment information
            all_segments_info = []
            for i, segment_data in enumerate(segments):
                segment_output_dir = os.path.join(segments_dir, f"segment_{i:03d}")
                os.makedirs(segment_output_dir, exist_ok=True)
                
                segment_info = {
                    'segment_id': i,
                    'start_frame_idx': segment_data['start_idx'],
                    'end_frame_idx': segment_data['end_idx'],
                    'length_frames': segment_data['length'],
                }
                all_segments_info.append(segment_info)
                
                # Save segment metadata
                with open(os.path.join(segment_output_dir, 'segment_info.yaml'), 'w') as f:
                    yaml.dump(segment_info, f)
                    
                # Save detections for this segment
                segment_detections = segment_data['detections']
                with open(os.path.join(segment_output_dir, 'detections.pkl'), 'wb') as f:
                    pickle.dump(segment_detections, f)

                # Save preview frames for the segment
                if config.get('detection', {}).get('visualization', False):
                    preview_dir = os.path.join(segment_output_dir, 'preview')
                    os.makedirs(preview_dir, exist_ok=True)
                    indices_to_save = [0, segment_data['length'] // 2, segment_data['length'] - 1]
                    for frame_offset in indices_to_save:
                        global_frame_idx = segment_data['start_idx'] + frame_offset
                        if 0 <= global_frame_idx < len(frames) and 0 <= frame_offset < len(segment_detections):
                            frame = frames[global_frame_idx]
                            dets = segment_detections[frame_offset]
                            viz_frame = detector.visualize_detections(frame, dets)
                            cv2.imwrite(os.path.join(preview_dir, f"frame_{frame_offset:04d}.jpg"), viz_frame)
            
            # Save summary of all segments found
            with open(os.path.join(video_output_dir, 'segments_summary.yaml'), 'w') as f:
                yaml.dump(all_segments_info, f)
                
        except Exception as e:
            logger.error(f"Error during segment filtering for {video_path}: {e}", exc_info=True)
            return # Stop processing this video
    
    # --- Step 4: Person Tracking within each segment ---
    if skip_to_step in ['pose', 'audio', 'integration']:
        logger.info("Skipping tracking step as requested")
    else:
        try:
            logger.info("Processing person tracking for segments")
            
            for i, segment_data in enumerate(segments):
                segment_output_dir = os.path.join(segments_dir, f"segment_{i:03d}")
                tracking_output_dir = os.path.join(segment_output_dir, 'tracking')
                tracking_viz_dir = os.path.join(segment_output_dir, 'visualization', 'tracking')
                os.makedirs(tracking_output_dir, exist_ok=True)
                os.makedirs(tracking_viz_dir, exist_ok=True)
                
                # Check if tracking results already exist
                tracking_file = os.path.join(tracking_output_dir, 'tracked_segment.pkl')
                if no_overwrite and os.path.exists(tracking_file):
                    logger.info(f"Skipping already tracked segment {i}")
                    continue
                
                # Process tracking for this segment
                logger.info(f"Tracking persons in segment {i}: frames {segment_data['start_idx']}-{segment_data['end_idx']}")
                tracked_segment = tracker.track_segment(segment_data, frames)
                
                # Save tracked segment
                with open(tracking_file, 'wb') as f:
                    pickle.dump(tracked_segment, f)
                
                # Save tracking metadata
                tracking_meta = {
                    'segment_id': i,
                    'start_frame_idx': tracked_segment['start_idx'],
                    'end_frame_idx': tracked_segment['end_idx'],
                    'length_frames': tracked_segment['length'],
                    'num_tracks': len(tracked_segment.get('tracks', [])),
                    'tracks': tracked_segment.get('tracks', []),
                }
                with open(os.path.join(tracking_output_dir, 'tracking_info.yaml'), 'w') as f:
                    yaml.dump(tracking_meta, f)
                
                # Save tracking visualization if enabled
                if config.get('tracking', {}).get('visualization', False):
                    # Create visualization frames
                    viz_frames = tracker.create_track_visualization(tracked_segment, frames)
                    
                    # Save visualization frames
                    for frame_idx, viz_frame in enumerate(viz_frames):
                        cv2.imwrite(os.path.join(tracking_viz_dir, f"frame_{frame_idx:04d}.jpg"), viz_frame)
                    
                    # Save preview frames (beginning, middle, end)
                    preview_dir = os.path.join(segment_output_dir, 'preview')
                    os.makedirs(preview_dir, exist_ok=True)
                    
                    indices_to_save = [0, len(viz_frames) // 2, len(viz_frames) - 1]
                    for idx, frame_idx in enumerate(indices_to_save):
                        if 0 <= frame_idx < len(viz_frames):
                            preview_file = os.path.join(preview_dir, f"tracking_{idx}.jpg")
                            cv2.imwrite(preview_file, viz_frames[frame_idx])
                
                # Extract person tracks (group by identity)
                person_tracks = tracker.get_person_tracks(tracked_segment)
                
                # Save individual person track data
                person_tracks_dir = os.path.join(tracking_output_dir, 'person_tracks')
                os.makedirs(person_tracks_dir, exist_ok=True)
                
                for track_id, track_detections in person_tracks.items():
                    track_file = os.path.join(person_tracks_dir, f"person_{track_id}.pkl")
                    with open(track_file, 'wb') as f:
                        pickle.dump(track_detections, f)
                
                logger.info(f"Completed tracking for segment {i}: found {len(person_tracks)} person tracks")
                
            logger.info("Successfully completed person tracking for all segments")
                
        except Exception as e:
            logger.error(f"Error during person tracking for {video_path}: {e}", exc_info=True)
    
    # --- Step 5: Motion Estimation (Body, Hand, Face) for each track ---
    if skip_to_step in ['audio', 'integration']:
        logger.info("Skipping pose estimation step as requested")
    else:
        try:
            logger.info("Processing pose estimation for segments")
            
            for i, segment_data in enumerate(segments):
                segment_output_dir = os.path.join(segments_dir, f"segment_{i:03d}")
                tracking_output_dir = os.path.join(segment_output_dir, 'tracking')
                pose_output_dir = os.path.join(segment_output_dir, 'motion')
                pose_viz_dir = os.path.join(segment_output_dir, 'visualization', 'pose')
                
                os.makedirs(pose_output_dir, exist_ok=True)
                os.makedirs(pose_viz_dir, exist_ok=True)
                
                # Check if tracked segment exists
                tracking_file = os.path.join(tracking_output_dir, 'tracked_segment.pkl')
                if not os.path.exists(tracking_file):
                    logger.warning(f"Tracked segment file not found for segment {i}, skipping pose estimation")
                    continue
                    
                # Check if pose results already exist
                pose_file = os.path.join(pose_output_dir, 'pose_results.pkl')
                if no_overwrite and os.path.exists(pose_file):
                    logger.info(f"Skipping already processed pose estimation for segment {i}")
                    continue
                    
                # Load tracked segment
                with open(tracking_file, 'rb') as f:
                    tracked_segment = pickle.load(f)
                    
                # Process pose estimation for this segment
                logger.info(f"Processing pose estimation for segment {i}: frames {tracked_segment['start_idx']}-{tracked_segment['end_idx']}")
                segment_with_pose = pose_estimator.process_segment(tracked_segment, frames)
                
                # Save pose results
                with open(pose_file, 'wb') as f:
                    pickle.dump(segment_with_pose, f)
                    
                # Save pose metadata
                pose_results = segment_with_pose.get('pose_results', {})
                pose_meta = {
                    'segment_id': i,
                    'start_frame_idx': segment_with_pose['start_idx'],
                    'end_frame_idx': segment_with_pose['end_idx'],
                    'length_frames': segment_with_pose['length'],
                    'num_tracks': len(pose_results),
                    'tracks': [{'id': track_id, 'num_frames': len(result.frame_indices)} 
                             for track_id, result in pose_results.items()]
                }
                with open(os.path.join(pose_output_dir, 'pose_info.yaml'), 'w') as f:
                    yaml.dump(pose_meta, f)
                    
                # Save individual SMPLX parameters for each track
                for track_id, result in pose_results.items():
                    track_dir = os.path.join(pose_output_dir, f"person_{track_id}")
                    os.makedirs(track_dir, exist_ok=True)
                    
                    # Save SMPLX parameters
                    smplx_file = os.path.join(track_dir, 'smplx_params.pkl')
                    with open(smplx_file, 'wb') as f:
                        pickle.dump(result.smplx_params, f)
                        
                    # Save 3D joints
                    joints_file = os.path.join(track_dir, 'joints_3d.npy')
                    np.save(joints_file, result.joints_3d)
                    
                    # Save frame mapping
                    frames_file = os.path.join(track_dir, 'frame_mapping.pkl')
                    with open(frames_file, 'wb') as f:
                        pickle.dump({
                            'local_indices': result.frame_indices,
                            'global_indices': result.global_frame_indices,
                            'confidence': result.confidence
                        }, f)
                
                # Save pose visualization if enabled
                if config.get('motion', {}).get('body', {}).get('visualization', False):
                    # Create visualization frames
                    save_pose_visualizations(frames, pose_estimator, segment_with_pose, pose_viz_dir, logger)
                    
                    # Save preview frames (beginning, middle, end)
                    preview_dir = os.path.join(segment_output_dir, 'preview')
                    os.makedirs(preview_dir, exist_ok=True)
                    
                    viz_frames = pose_estimator.create_pose_visualization(segment_with_pose, frames)
                    if viz_frames:
                        indices_to_save = [0, len(viz_frames) // 2, len(viz_frames) - 1]
                        for idx, frame_idx in enumerate(indices_to_save):
                            if 0 <= frame_idx < len(viz_frames):
                                preview_file = os.path.join(preview_dir, f"pose_{idx}.jpg")
                                cv2.imwrite(preview_file, viz_frames[frame_idx])
                
                logger.info(f"Completed pose estimation for segment {i}: processed {len(pose_results)} person tracks")
                
            logger.info("Successfully completed pose estimation for all segments")
                
        except Exception as e:
            logger.error(f"Error during pose estimation for {video_path}: {e}", exc_info=True)
    
    # --- TODO: Implement Audio Processing --- 
    if skip_to_step not in ['integration']:
        logger.info("--- Placeholder for Audio Processing ---")
    # Step 6: Audio Processing (Speaker Diarization, Separation)
    # Step 7: Data Integration and final export
    
    # --- Mark processing as complete --- 
    with open(completion_flag, 'w') as f:
        f.write(datetime.datetime.now().isoformat())
        
    logger.info(f"Successfully completed processing for {os.path.basename(video_path)}")

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # Set up logger
    logger = setup_logger(config)
    
    # Log configuration
    log_config(logger, config)
    
    # Create directories
    setup_directories(config)
    
    # Check CUDA availability
    try:
        device = config['general']['device']
        if 'cuda' in device:
            if not torch.cuda.is_available():
                logger.warning(f"CUDA device '{device}' requested but not available. Falling back to CPU.")
                config['general']['device'] = 'cpu'
            else:
                # Try setting device to check validity
                torch.cuda.get_device_name(device)
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        else:
            logger.info("Using CPU device")
    except Exception as e:
        logger.warning(f"Error setting device '{config['general']['device']}'. Falling back to CPU. Error: {e}")
        config['general']['device'] = 'cpu'
        # Update device in sub-configs as well
        if 'device' in config['detection']:
             config['detection']['device'] = 'cpu'
        # Add for other modules (motion, etc.) as needed

    # Get video files
    video_files = get_video_files(args.input_dir)
    logger.info(f"Found {len(video_files)} video files in {args.input_dir}")
    
    # Limit number of videos if specified
    if args.max_videos is not None and args.max_videos > 0:
        video_files = video_files[:args.max_videos]
        logger.info(f"Processing only the first {args.max_videos} videos")
    
    # Process each video
    processed_count = 0
    error_count = 0
    skipped_count = 0 # Add counter for skipped videos

    for i, video_file in enumerate(video_files):
        try:
            logger.info(f"--- Processing video {i+1}/{len(video_files)}: {os.path.basename(video_file)} ---")
            # Check skip condition before calling process_video
            video_output_dir = get_output_path_for_video(video_file, config['general']['output_dir'])
            completion_flag = os.path.join(video_output_dir, 'processing_complete.flag')
            if args.no_overwrite and os.path.exists(completion_flag):
                 logger.info(f"Skipping already completed video: {video_file}")
                 skipped_count += 1
                 continue
                 
            process_video(video_file, config, logger, args.no_overwrite, args.skip_to_step)
            processed_count += 1
        except Exception as e:
            logger.error(f"Unhandled error processing {video_file}: {str(e)}", exc_info=True)
            error_count += 1
    
    logger.info(f"--- Processing Summary ---")
    logger.info(f"Total videos found: {len(video_files)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped (already complete): {skipped_count}")
    logger.info(f"Errors encountered: {error_count}")
    logger.info(f"Results saved to: {config['general']['output_dir']}")
    logger.info("Processing complete.")

if __name__ == "__main__":
    main() 