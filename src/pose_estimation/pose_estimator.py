"""
Pose Estimation module for body poses.

This module contains the PoseEstimator class which integrates
the 4D-Humans/HMR2 model for body pose estimation. It processes person
detections and extracts SMPLX pose parameters for each person.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import cv2
from dataclasses import dataclass
from collections import defaultdict

# Attempt to import the actual HMR2 library
try:
    # Import path based on common library structures and package name `hmr2`
    from hmr2.models import HMR2 # Assuming this is the correct model class
    HMR2_AVAILABLE = True
except ImportError:
    HMR2_AVAILABLE = False
    logging.warning("HMR2 (4D-Humans) library not found or import failed. Pose estimation will not work.")
    # Define a placeholder class if the real one isn't available
    class HMR2:
        def __init__(*args, **kwargs):
            pass
        def __call__(*args, **kwargs):
             # Mock output structure similar to what HMR2 might provide
             # It's important to adjust this based on actual HMR2 output if known
             mock_batch_size = 1 if not args or not hasattr(args[1], 'shape') else args[1].shape[0]
             return {
                 'pred_smpl_params': {
                     'global_orient': np.zeros((mock_batch_size, 1, 3, 3)),
                     'body_pose': np.zeros((mock_batch_size, 21, 3, 3)),
                     'betas': np.zeros((mock_batch_size, 10)),
                 },
                 'pred_cam': np.zeros((mock_batch_size, 3)),
                 'pred_joints': np.zeros((mock_batch_size, 22, 3)) # Example: 22 body joints
             }
        # Add a placeholder visualize_pose if needed, or rely on default processing
        def visualize_pose(self, image, *args, **kwargs):
            logging.warning("HMR2 visualize_pose not available (mock). Returning original image.")
            return image


@dataclass
class PoseEstimationResult:
    """Class to store pose estimation results for a single person track."""
    
    track_id: int
    smplx_params: Dict[str, np.ndarray]  # SMPLX parameters (or subset from HMR2)
    joints_3d: np.ndarray  # 3D joint positions
    pred_cam: Optional[np.ndarray] # Predicted camera parameters
    confidence: np.ndarray  # Confidence scores for each frame
    frame_indices: List[int]  # Local frame indices in the segment
    global_frame_indices: List[int]  # Global frame indices in the original video
    

class PoseEstimator:
    """Class to handle body pose estimation using 4D-Humans/HMR2 model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the PoseEstimator.
        
        Args:
            config: Configuration dictionary for pose estimation
        """
        self.logger = logging.getLogger('PoseEstimator')
        
        # Parse configuration
        self.model_path = config.get('model_path', 'models/hmr2/model.pt') # Adjusted default path potentially
        self.batch_size = config.get('batch_size', 8)
        self.temporal_smoothing = config.get('temporal_smoothing', True)
        self.smoothing_window = config.get('smoothing_window', 5)
        self.visualization = config.get('visualization', False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the model
        if not HMR2_AVAILABLE:
            self.logger.error("HMR2 library is not installed or import failed. Cannot initialize PoseEstimator.")
            self.model = None
            return
            
        if not os.path.exists(self.model_path):
             self.logger.error(f"HMR2 model checkpoint not found at {self.model_path}")
             self.model = None
             return
             
        try:
            self.logger.info(f"Initializing HMR2 model from {self.model_path} on device {self.device}")
            # Instantiate the actual HMR2 model
            # Arguments might need adjustment based on the actual HMR2 API
            # Example: HMR2.load_from_checkpoint(self.model_path)
            # For now, assume a simple constructor or a specific load method
            # This part likely needs specific knowledge of the HMR2 library usage
            # Let's assume a constructor for now:
            self.model = HMR2(pretrained_ckpt=self.model_path) # Example instantiation
            self.model.eval()
            self.model.to(self.device)
            
            self.logger.info("PoseEstimator initialized successfully with HMR2 model")
        except Exception as e:
            self.logger.error(f"Failed to initialize HMR2 model: {e}", exc_info=True)
            self.model = None
        
    def crop_person(self, frame: np.ndarray, bbox: List[float], padding: float = 0.1) -> np.ndarray:
        """Crop a person from a frame using the bounding box.
        
        Args:
            frame: Video frame
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Optional padding ratio for the crop
            
        Returns:
            Cropped person image, resized for model input (e.g., 224x224)
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Add padding
        h, w = frame.shape[:2]
        width, height = x2 - x1, y2 - y1
        
        # Calculate padding
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        # Crop the image
        crop = frame[y1:y2, x1:x2].copy()
        
        # Resize crop to expected model input size (e.g., 224x224 for HMR2)
        # This size might need to be configurable or determined from the model
        target_size = (224, 224)
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            # Handle cases where crop is empty
            crop_resized = np.zeros((target_size[1], target_size[0], 3), dtype=frame.dtype)
            self.logger.warning("Encountered empty crop after padding/bounding box extraction.")

        return crop_resized
        
    def preprocess_batch(self, crops: List[np.ndarray]) -> Optional[torch.Tensor]:
        """Preprocess a batch of images for HMR2 input.
        Assumes normalization and conversion to torch tensor.
        This might need specifics from HMR2 documentation.
        """
        if not crops:
            return None
        # Example preprocessing: Normalize and convert to tensor
        # Normalization values often depend on the model training (e.g., ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        
        batch = []
        for img in crops:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().to(self.device) / 255.0
            batch.append(img_tensor)
            
        batch_tensor = torch.stack(batch)
        batch_tensor = (batch_tensor - mean) / std
        return batch_tensor

    def process_batch(self, crops: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Process a batch of person crops for pose estimation using HMR2.
        
        Args:
            crops: List of pre-resized person crop images (e.g., 224x224)
            
        Returns:
            HMR2 model output dictionary or None if processing fails.
        """
        if not crops or self.model is None:
            return None

        # Preprocess the batch
        batch_tensor = self.preprocess_batch(crops)
        if batch_tensor is None:
             return None

        # Process crops through the HMR2 model
        try:
            with torch.no_grad():
                results = self.model(batch_tensor)
                
            # Convert results to numpy arrays, detaching from graph
            numpy_results = {}
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, torch.Tensor):
                        numpy_results[key] = value.cpu().numpy()
                    elif isinstance(value, dict): # Handle nested dicts like pred_smpl_params
                        numpy_results[key] = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in value.items()}
                    else:
                        numpy_results[key] = value
            else:
                 self.logger.error("HMR2 model output was not a dictionary as expected.")
                 return None
                 
        except Exception as e:
            self.logger.error(f"Error during HMR2 inference: {e}", exc_info=True)
            return None
        
        return numpy_results
        
    def apply_temporal_smoothing(self, params_dict: Dict[str, np.ndarray], window_size: int = 5) -> Dict[str, np.ndarray]:
        """Apply temporal smoothing to pose parameters (e.g., SMPL parameters).
        Operates on dictionaries where values are numpy arrays [num_frames, ...].
        """
        smoothed_params = {}
        
        if not params_dict:
            return {}
            
        # Get the number of frames
        num_frames = 0
        for param_values in params_dict.values():
            if isinstance(param_values, np.ndarray) and param_values.ndim >= 1:
                num_frames = param_values.shape[0]
                break
        
        # Skip smoothing if we have too few frames
        if num_frames < window_size:
            return params_dict
            
        # Apply smoothing to each parameter array in the dictionary
        for param_name, param_values in params_dict.items():
            # Only smooth numpy arrays with a temporal dimension
            if not isinstance(param_values, np.ndarray) or param_values.ndim < 1 or param_values.shape[0] != num_frames:
                smoothed_params[param_name] = param_values
                continue
                
            # Apply a simple moving average filter along the temporal dimension (axis=0)
            kernel = np.ones(window_size) / window_size
            
            # Handle different dimensions
            if param_values.ndim == 1:
                 smoothed_values = np.convolve(param_values, kernel, mode='same')
                 # Boundary handling
                 half_window = window_size // 2
                 smoothed_values[:half_window] = param_values[:half_window]
                 smoothed_values[-half_window:] = param_values[-half_window:]
            else:
                # For multi-dimensional arrays, apply smoothing along axis 0
                smoothed_values = np.apply_along_axis(
                    lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=param_values
                )
                # Boundary handling
                half_window = window_size // 2
                smoothed_values[:half_window] = param_values[:half_window]
                smoothed_values[-half_window:] = param_values[-half_window:]
            
            smoothed_params[param_name] = smoothed_values
            
        return smoothed_params
    
    def process_track(self, track_id: int, track_data: Dict[str, Any], frames: List[np.ndarray]) -> Optional[PoseEstimationResult]:
        """Process a single person track for pose estimation.
        
        Args:
            track_id: ID of the track
            track_data: Track data containing bounding boxes and frame indices
            frames: List of video frames
            
        Returns:
            PoseEstimationResult for the track, or None if processing fails
        """
        if self.model is None:
            self.logger.error("PoseEstimator model is not initialized. Cannot process track.")
            return None
            
        self.logger.info(f"Processing pose estimation for track {track_id} with {len(track_data['frame_indices'])} detections")
        
        # Extract data from the track
        bboxes = track_data['bboxes']
        frame_indices = track_data['frame_indices'] # Local frame indices
        global_frame_indices_map = track_data['global_frame_indices'] # Global frame indices
        
        # Skip tracks with too few detections
        if len(frame_indices) < 3:
            self.logger.warning(f"Skipping track {track_id} with only {len(frame_indices)} detections")
            return None
            
        # Prepare crops and corresponding global indices
        crops = []
        valid_frame_indices = [] # Local indices corresponding to valid crops
        valid_global_indices = [] # Global indices corresponding to valid crops
        original_indices_for_valid_crops = [] # Original indices within the input lists (frame_indices, bboxes)
        
        for i, (local_idx, global_idx, bbox) in enumerate(zip(frame_indices, global_frame_indices_map, bboxes)):
            if 0 <= global_idx < len(frames):
                crop = self.crop_person(frames[global_idx], bbox)
                if crop is not None and crop.size > 0:
                    crops.append(crop)
                    valid_frame_indices.append(local_idx)
                    valid_global_indices.append(global_idx)
                    original_indices_for_valid_crops.append(i)
                else:
                    self.logger.warning(f"Failed to create valid crop for frame {global_idx} in track {track_id}")
            else:
                self.logger.warning(f"Invalid global frame index {global_idx} for track {track_id}, skipping this frame")
        
        if not crops:
             self.logger.warning(f"No valid crops found for track {track_id}")
             return None

        # Process in batches
        all_pred_smpl_params = defaultdict(list)
        all_pred_cam = []
        all_pred_joints = []
        
        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i:i+self.batch_size]
            
            batch_results = self.process_batch(batch_crops)
            
            if batch_results and 'pred_smpl_params' in batch_results: 
                # Append batch results to overall results
                # HMR2 nests params under 'pred_smpl_params'
                pred_params = batch_results['pred_smpl_params']
                for param_name, param_values in pred_params.items():
                    all_pred_smpl_params[param_name].append(param_values)
                
                if 'pred_cam' in batch_results:
                    all_pred_cam.append(batch_results['pred_cam'])
                if 'pred_joints' in batch_results:
                    all_pred_joints.append(batch_results['pred_joints'])
            else:
                self.logger.warning(f"Pose estimation failed or returned invalid format for batch starting at index {i} for track {track_id}")
                # Handle failed batch - e.g., insert NaNs or skip? For now, size mismatch will occur.
                # Need a strategy here - maybe return None for the whole track if any batch fails?
                return None # Fail track if any batch fails for simplicity
                
        if not all_pred_joints:
            self.logger.error(f"Pose estimation failed for all batches in track {track_id}")
            return None

        # Concatenate results from all batches
        pred_smpl_params = {param_name: np.concatenate(param_values) 
                            for param_name, param_values in all_pred_smpl_params.items() if param_values}
        pred_cam = np.concatenate(all_pred_cam) if all_pred_cam else None
        pred_joints = np.concatenate(all_pred_joints) if all_pred_joints else None
        
        # Apply temporal smoothing if enabled
        if self.temporal_smoothing and len(crops) > self.smoothing_window:
            # Smooth only the relevant SMPL parameters (pose, orient), not betas
            params_to_smooth = {k: v for k, v in pred_smpl_params.items() if k != 'betas'}
            smoothed_subset = self.apply_temporal_smoothing(params_to_smooth, self.smoothing_window)
            pred_smpl_params.update(smoothed_subset)
            # Optionally smooth camera or joints if needed
            # pred_cam = self.apply_temporal_smoothing({'cam': pred_cam}, self.smoothing_window)['cam']
            # pred_joints = self.apply_temporal_smoothing({'joints': pred_joints}, self.smoothing_window)['joints']

        # Confidence - HMR2 doesn't typically output per-frame confidence easily.
        # Use a placeholder or derive from reprojection error if available/calculated.
        confidence = np.ones(len(valid_frame_indices)) # Placeholder confidence
        
        # Create result object
        result = PoseEstimationResult(
            track_id=track_id,
            smplx_params=pred_smpl_params, # Note: HMR2 outputs SMPL params, map to SMPLX if needed later
            joints_3d=pred_joints, # Use the predicted joints from HMR2
            pred_cam=pred_cam,
            confidence=confidence,
            frame_indices=valid_frame_indices,  
            global_frame_indices=valid_global_indices
        )
        
        self.logger.info(f"Completed pose estimation for track {track_id}")
        return result
        
    def process_segment(self, segment: Dict[str, Any], frames: List[np.ndarray]) -> Dict[str, Any]:
        """Process all person tracks in a segment.
        
        Args:
            segment: Tracked segment containing multiple person tracks
            frames: List of video frames
            
        Returns:
            Segment dictionary with added pose results
        """
        if self.model is None:
            self.logger.error("PoseEstimator model is not initialized. Cannot process segment.")
            segment['pose_results'] = {}
            return segment
            
        # Extract tracks dictionary correctly (adjust based on actual tracking output format)
        person_tracks_data = segment.get('person_tracks') # Assuming get_person_tracks was called
        if not person_tracks_data:
             # Try to get it from the tracked_segment structure if not pre-extracted
             try:
                 temp_tracker = PersonTracker({}) # Temp tracker instance to use the extraction method
                 person_tracks_data = temp_tracker.get_person_tracks(segment)
             except Exception:
                 person_tracks_data = None
                 
        if not person_tracks_data or not isinstance(person_tracks_data, dict):
            self.logger.warning(f"No valid person tracks found in segment {segment.get('segment_id', '?')}. Skipping pose estimation.")
            segment['pose_results'] = {}
            return segment
            
        # Create a copy of the segment to add pose results
        segment_with_pose = segment.copy()
        segment_with_pose['pose_results'] = {}
        
        # Reconstruct track data in the format process_track expects
        reconstructed_tracks = {}
        for track_id, detections in person_tracks_data.items():
            if not detections: continue
            reconstructed_tracks[track_id] = {
                 'bboxes': [d['bbox'] for d in detections],
                 'frame_indices': [d['frame_offset'] for d in detections],
                 'global_frame_indices': [d['frame_idx'] for d in detections]
            }

        # Process each track
        for track_id, track_data in reconstructed_tracks.items():
            # Process pose for this track
            pose_result = self.process_track(track_id, track_data, frames)
            
            if pose_result:
                segment_with_pose['pose_results'][track_id] = pose_result
                
        num_processed = len(segment_with_pose['pose_results'])
        self.logger.info(f"Completed pose estimation for segment {segment.get('segment_id', '?')}: {num_processed} tracks processed.")
        return segment_with_pose
        
    def create_pose_visualization(self, segment_with_pose: Dict[str, Any], frames: List[np.ndarray]) -> List[np.ndarray]:
        """Create visualization frames for pose estimation results.
        
        Args:
            segment_with_pose: Segment with added pose results
            frames: List of video frames
            
        Returns:
            List of visualization frames
        """
        if not self.visualization or self.model is None or not hasattr(self.model, 'visualize_pose'): # Check if model or visualize method exists
            return []
            
        pose_results = segment_with_pose.get('pose_results', {})
        if not pose_results:
            self.logger.warning("No pose results found in segment, skipping visualization")
            return []
            
        # Extract frame range for the segment
        start_idx = segment_with_pose.get('start_idx', -1)
        end_idx = segment_with_pose.get('end_idx', -1)
        
        if start_idx < 0 or end_idx < 0 or end_idx >= len(frames) or start_idx > end_idx:
            self.logger.warning(f"Invalid frame range: {start_idx}-{end_idx}, skipping visualization")
            return []
            
        # Create a copy of frames for visualization
        segment_frames = frames[start_idx : end_idx + 1]
        viz_frames = [frame.copy() for frame in segment_frames]
        
        # Create dictionary mapping relative frame index to track poses
        frame_to_poses = defaultdict(list)
        
        # Collect poses for each frame
        for track_id, pose_result in pose_results.items():
            if not isinstance(pose_result, PoseEstimationResult): continue
            
            for i, global_frame_idx in enumerate(pose_result.global_frame_indices):
                rel_frame_idx = global_frame_idx - start_idx
                if 0 <= rel_frame_idx < len(viz_frames):
                    # Extract the parameters for this specific frame
                    # Adapt based on HMR2 output structure in pose_result.smplx_params
                    frame_smpl_params = {p: v[i:i+1] for p, v in pose_result.smplx_params.items() if hasattr(v, 'shape') and v.ndim > 0 and v.shape[0] == len(pose_result.global_frame_indices)}
                    frame_joints = pose_result.joints_3d[i:i+1] if pose_result.joints_3d.ndim > 1 and pose_result.joints_3d.shape[0] == len(pose_result.global_frame_indices) else None
                    pred_cam = pose_result.pred_cam[i:i+1] if pose_result.pred_cam is not None and pose_result.pred_cam.shape[0] == len(pose_result.global_frame_indices) else None
                    
                    if frame_smpl_params and frame_joints is not None:
                        frame_to_poses[rel_frame_idx].append({
                            'track_id': track_id,
                            'smpl_params': frame_smpl_params,
                            'joints_3d': frame_joints,
                            'pred_cam': pred_cam
                        })
        
        # Visualize poses on each frame
        for rel_frame_idx, poses_in_frame in frame_to_poses.items():
            current_viz_frame = viz_frames[rel_frame_idx]
            for pose_data in poses_in_frame:
                # Visualize the pose using the model's visualizer
                # The visualize_pose method might need different arguments for HMR2
                try:
                    # Example: visualize_pose might take the image and the full output dict
                    # This is highly dependent on the actual HMR2 visualizer API
                    render_output = { # Reconstruct a dict similar to model output for one frame
                         'pred_smpl_params': pose_data['smpl_params'],
                         'pred_joints': pose_data['joints_3d'],
                         'pred_cam': pose_data['pred_cam']
                    }
                    # We might need to pass the original crop or bbox info as well
                    current_viz_frame = self.model.visualize_pose(current_viz_frame, render_output) 
                except Exception as e:
                    self.logger.warning(f"Error visualizing pose for track {pose_data['track_id']} on frame {start_idx + rel_frame_idx}: {e}")
                    # Draw a simple marker if visualization fails
                    cv2.putText(current_viz_frame, f"ID: {pose_data['track_id']} (Viz Error)", (10, 60 + pose_data['track_id']*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue
                    
                # Add track ID (if not added by visualizer)
                # cv2.putText(current_viz_frame, f"ID: {pose_data['track_id']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update the visualization frame list
            viz_frames[rel_frame_idx] = current_viz_frame
        
        return viz_frames 