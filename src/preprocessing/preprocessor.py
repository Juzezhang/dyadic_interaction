"""
Video preprocessing module for the Dyadic Interaction Dataset Generator.

This module handles:
- Frame extraction from videos
- Scene change detection
- Frame normalization (resizing, optional color correction)
- Metadata extraction
- Caching of preprocessed data
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """Video preprocessing class for extracting and normalizing frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing parameters
                - fps: Target frames per second
                - resolution: Target resolution [height, width]
                - scene_change_threshold: Threshold for scene change detection
                - cache_frames: Boolean flag to enable/disable caching
                - cache_dir: Directory for caching preprocessed frames (inherited from general config if not specified)
                - color_normalize: Boolean flag for color normalization (optional)
        """
        self.fps = config.get('fps', 30)
        self.resolution = tuple(config.get('resolution', [720, 1280]))
        self.scene_change_threshold = config.get('scene_change_threshold', 35.0)
        self.cache_frames = config.get('cache_frames', True)
        self.cache_dir = config.get('cache_dir', 'data/cache') # This should ideally be set in main.py from general config
        self.color_normalize = config.get('color_normalize', False)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Initialized VideoPreprocessor with fps={self.fps}, "
                    f"resolution={self.resolution}, caching={'enabled' if self.cache_frames else 'disabled'}")

    def _get_cache_path(self, video_path: str) -> str:
        """Generate cache file path for a video."""
        video_filename = os.path.basename(video_path)
        cache_filename = f"{os.path.splitext(video_filename)[0]}_fps{self.fps}_res{self.resolution[0]}x{self.resolution[1]}.json"
        return os.path.join(self.cache_dir, cache_filename)

    def _load_from_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """Load preprocessing results from cache file."""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded preprocessed data from cache: {cache_path}")
                # Note: Frames themselves are not cached, only metadata and scene changes
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_path}: {e}")
        return None

    def _save_to_cache(self, cache_path: str, data: Dict[str, Any]) -> None:
        """Save preprocessing results (metadata, scene changes) to cache file."""
        try:
            # Don't save frames in the JSON cache
            cacheable_data = data.copy()
            if 'frames' in cacheable_data:
                del cacheable_data['frames']
                
            with open(cache_path, 'w') as f:
                json.dump(cacheable_data, f, indent=4)
            logger.info(f"Saved preprocessing metadata to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_path}: {e}")

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video file to extract frames and detect scene changes.
        Uses caching if enabled.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing:
                - frames: List of extracted frame arrays (numpy ndarray, BGR format)
                - scene_changes: List of frame indices where scenes change
                - metadata: Video metadata
        """
        logger.info(f"Processing video: {video_path}")
        cache_path = self._get_cache_path(video_path)
        
        # Try loading from cache
        if self.cache_frames:
            cached_data = self._load_from_cache(cache_path)
            if cached_data:
                # Need to re-extract frames even if metadata is cached
                logger.info("Cache hit for metadata. Re-extracting frames...")
                frames, _ = self._extract_frames(video_path, cached_data['metadata'], use_tqdm=True)
                cached_data['frames'] = frames
                return cached_data
        
        logger.info("Cache miss or caching disabled. Processing video from scratch.")
        # Extract metadata
        metadata = self._extract_metadata(video_path)
        if not metadata:
             return { 'frames': [], 'scene_changes': [], 'metadata': {'error': 'Failed to extract metadata'} }

        # Extract frames and detect scene changes
        frames, scene_changes = self._extract_frames(video_path, metadata, use_tqdm=True)
        
        logger.info(f"Extracted {len(frames)} frames with {len(scene_changes)} scene changes")
        
        results = {
            'frames': frames,
            'scene_changes': scene_changes,
            'metadata': metadata
        }
        
        # Save metadata and scene changes to cache
        if self.cache_frames:
            self._save_to_cache(cache_path, results)
            
        return results
    
    def _extract_metadata(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Extract metadata from video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return None
            
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / original_fps if original_fps > 0 else 0
            
            cap.release()
            
            return {
                'original_fps': original_fps,
                'frame_count': frame_count,
                'original_width': width,
                'original_height': height,
                'duration': duration,
                'path': video_path,
                'filename': os.path.basename(video_path),
                'target_fps': self.fps,
                'target_resolution': self.resolution
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {video_path}: {e}")
            return None
    
    def _extract_frames(self, video_path: str, metadata: Dict[str, Any], use_tqdm: bool = False) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract frames from video, normalize, and detect scene changes.
        
        Args:
            video_path: Path to the video file
            metadata: Video metadata dictionary
            use_tqdm: Whether to display a progress bar
            
        Returns:
            Tuple containing:
                - List of extracted frame arrays (BGR format)
                - List of frame indices where scenes change (relative to extracted frames)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file during frame extraction: {video_path}")
            return [], []
        
        original_fps = metadata.get('original_fps', 30) # Default to 30 if missing
        target_fps = self.fps
        total_frames = metadata.get('frame_count', 0)
        
        # Calculate frame sampling interval
        if original_fps <= 0 or target_fps <= 0:
            interval = 1 # Avoid division by zero or negative fps
        else:
            interval = max(1, round(original_fps / target_fps))
        
        frames = []
        scene_changes = []
        prev_frame_processed = None
        frame_idx_original = 0
        extracted_frame_count = 0

        # Setup progress bar if requested
        progress_bar = None
        if use_tqdm and total_frames > 0:
            progress_bar = tqdm(total=total_frames, desc=f"Extracting frames ({self.fps} fps)", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
            
            # Sample frames according to target FPS
            if frame_idx_original % interval == 0:
                # Normalize frame
                frame_processed = self._normalize_frame(frame)
                
                # Detect scene change if we have a previous processed frame
                if prev_frame_processed is not None:
                    diff = self._frame_difference(prev_frame_processed, frame_processed)
                    if diff > self.scene_change_threshold:
                        # Index relative to the *extracted* frames list
                        scene_changes.append(extracted_frame_count) 
                
                frames.append(frame_processed)
                prev_frame_processed = frame_processed.copy() # Store a copy for diff calculation
                extracted_frame_count += 1
            
            frame_idx_original += 1
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
            
        cap.release()
        return frames, scene_changes
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize a single frame (resizing, optional color correction)."""
        # Resize frame
        frame = cv2.resize(frame, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)
        
        # Optional: Color normalization/correction (placeholder)
        if self.color_normalize:
            # Example: Simple histogram equalization on grayscale
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # equalized = cv2.equalizeHist(gray)
            # frame = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            pass # Implement desired color normalization here
            
        return frame

    def _frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate difference between two frames for scene change detection.
        Uses grayscale difference mean.
        Consider using more robust methods like PySceneDetect if needed.
        
        Args:
            frame1: First frame (already normalized)
            frame2: Second frame (already normalized)
            
        Returns:
            Difference score between frames
        """
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Calculate mean difference
            score = np.mean(diff)
        except cv2.error as e:
            logger.warning(f"OpenCV error calculating frame difference: {e}")
            score = 0.0 # Return 0 difference on error
            
        return score 