"""
Person detection module for the Dyadic Interaction Dataset Generator.

This module handles:
- Person detection in video frames
- Bounding box extraction
- Filtering for exactly two-person scenes
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class PersonDetector:
    """Person detection class to identify people in video frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the person detector with configuration.
        
        Args:
            config: Dictionary containing detection parameters
                - model: YOLO model version ('yolov8n.pt', 'yolov8s.pt', etc.)
                - confidence_threshold: Minimum confidence for detections
                - device: Device to run inference on ('cpu', 'cuda:0', etc.)
                - batch_size: Number of frames to process in a batch
                - min_bbox_area: Minimum area (in pixels) for valid detections
                - min_bbox_ratio: Minimum ratio of bbox area to frame area
                - min_person_height: Minimum height of person as ratio of frame height
                - max_bbox_ratio: Maximum ratio of bbox area to frame area
                - max_person_height: Maximum height ratio of person to frame
                - max_edge_proximity: Maximum proximity to frame edge
                - aspect_ratio_range: Valid range for width/height ratio
        """
        self.model_name = config.get('model', 'yolov8n.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.device = config.get('device', 'cuda:0')
        self.batch_size = config.get('batch_size', 16)
        # Store the full config for accessing other parameters
        self.config = config
        
        logger.info(f"Initializing PersonDetector with model={self.model_name}, "
                   f"confidence_threshold={self.confidence_threshold}, device={self.device}")
        
        # Load YOLO model
        self.model = YOLO(self.model_name)
        
    def detect_persons(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Detect persons in a list of video frames.
        
        Args:
            frames: List of video frames (numpy arrays)
            
        Returns:
            List of lists, where each inner list contains detection results for a frame
            Each detection is a dictionary with keys:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - class_id: Class ID (should be person class)
        """
        logger.info(f"Detecting persons in {len(frames)} frames")
        
        results = []
        
        # Process frames in batches
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i+self.batch_size]
            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(frames)-1)//self.batch_size + 1}")
            
            # Run inference
            batch_results = self.model(batch, verbose=False)
            
            # Extract person detections for each frame
            for frame_result in batch_results:
                frame_detections = []
                
                # Filter for person class (class 0 in COCO dataset)
                for box in frame_result.boxes:
                    if box.cls.cpu().numpy()[0] == 0:  # Person class
                        confidence = box.conf.cpu().numpy()[0]
                        
                        # Apply confidence threshold
                        if confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': 0  # Person class
                            }
                            frame_detections.append(detection)
                
                results.append(frame_detections)
        
        logger.info(f"Detection complete. Found persons in {sum(1 for r in results if len(r) > 0)} frames")
        return results
    
    def filter_two_person_segments(self, 
                                  frames: List[np.ndarray],
                                  detections: List[List[Dict[str, Any]]],
                                  scene_changes: List[int],
                                  min_segment_length: int = 30) -> List[Dict[str, Any]]:
        """
        Filter video to identify segments with exactly two people.
        
        Args:
            frames: List of video frames
            detections: List of detection results per frame
            scene_changes: List of frame indices where scenes change
            min_segment_length: Minimum number of frames for a valid segment
            
        Returns:
            List of dictionaries describing two-person segments:
                - start_idx: Starting frame index
                - end_idx: Ending frame index
                - length: Number of frames in segment
                - detections: List of detections for each frame in the segment
        """
        logger.info("Filtering for two-person segments")
        
        segments = []
        current_segment = None
        
        # Convert scene_changes to a set for faster lookup
        scene_change_set = set(scene_changes)
        
        # Get frame dimensions for calculating ratios
        if frames and len(frames) > 0:
            frame_height, frame_width = frames[0].shape[:2]
            frame_area = frame_height * frame_width
        else:
            frame_height, frame_width, frame_area = 720, 1280, 720*1280  # Default values
            
        # Get minimum size thresholds from config
        min_bbox_area = self.config.get('min_bbox_area', 10000)  # Default: 100x100 pixels
        min_bbox_ratio = self.config.get('min_bbox_ratio', 0.01)  # Default: 1% of frame
        min_person_height = self.config.get('min_person_height', 0.2)  # Default: 20% of frame height
        min_frame_area_threshold = frame_area * min_bbox_ratio
        min_height_threshold = frame_height * min_person_height
        
        # Get maximum size thresholds from config (for filtering partial faces/too close to camera)
        max_bbox_ratio = self.config.get('max_bbox_ratio', 0.4)  # Default: 40% of frame
        max_person_height = self.config.get('max_person_height', 0.9)  # Default: 90% of frame height
        max_edge_proximity = self.config.get('max_edge_proximity', 0.05)  # Default: 5% of frame dimension
        aspect_ratio_range = self.config.get('aspect_ratio_range', [0.3, 0.9])  # Default: width/height between 0.3 and 0.9
        
        max_frame_area_threshold = frame_area * max_bbox_ratio
        max_height_threshold = frame_height * max_person_height
        max_edge_dist_x = int(frame_width * max_edge_proximity)  # Max distance to left/right edge
        max_edge_dist_y = int(frame_height * max_edge_proximity)  # Max distance to top/bottom edge
        
        logger.debug(f"Frame dimensions: {frame_width}x{frame_height}, area: {frame_area}")
        logger.debug(f"Minimum thresholds - area: {min_bbox_area}, ratio: {min_bbox_ratio} (={min_frame_area_threshold} px²), height: {min_person_height} (={min_height_threshold} px)")
        logger.debug(f"Maximum thresholds - ratio: {max_bbox_ratio} (={max_frame_area_threshold} px²), height: {max_person_height} (={max_height_threshold} px), edge proximity: {max_edge_proximity}")
        logger.debug(f"Aspect ratio range: {aspect_ratio_range}")
        
        # Function to check if a detection is valid based on size and position
        def is_valid_detection(det):
            bbox = det['bbox']  # [x1, y1, x2, y2]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            # Check minimum area (absolute value)
            if area < min_bbox_area:
                logger.debug(f"Detection rejected: area {area} < min_bbox_area {min_bbox_area}")
                return False
                
            # Check minimum area (as percentage of frame)
            if area < min_frame_area_threshold:
                logger.debug(f"Detection rejected: area ratio {area/frame_area:.4f} < min_bbox_ratio {min_bbox_ratio}")
                return False
                
            # Check minimum height (as percentage of frame height)
            if height < min_height_threshold:
                logger.debug(f"Detection rejected: height ratio {height/frame_height:.4f} < min_person_height {min_person_height}")
                return False
                
            # Check maximum area (as percentage of frame) - too large means too close to camera
            if area > max_frame_area_threshold:
                logger.debug(f"Detection rejected: area ratio {area/frame_area:.4f} > max_bbox_ratio {max_bbox_ratio}")
                return False
                
            # Check maximum height (as percentage of frame height)
            if height > max_height_threshold:
                logger.debug(f"Detection rejected: height ratio {height/frame_height:.4f} > max_person_height {max_person_height}")
                return False
                
            # Check aspect ratio (width/height ratio)
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                logger.debug(f"Detection rejected: aspect ratio {aspect_ratio:.4f} outside valid range {aspect_ratio_range}")
                return False
                
            # Check if detection is too close to frame edge (possibly cut off)
            if bbox[0] < max_edge_dist_x or bbox[2] > frame_width - max_edge_dist_x:
                logger.debug(f"Detection rejected: too close to left/right edge")
                return False
                
            if bbox[1] < max_edge_dist_y or bbox[3] > frame_height - max_edge_dist_y:
                logger.debug(f"Detection rejected: too close to top/bottom edge")
                return False
            
            return True
        
        for i, frame_detections in enumerate(detections):
            # Filter to keep only valid detections (large enough)
            valid_detections = [det for det in frame_detections if is_valid_detection(det)]
            
            # Check if current frame has exactly two VALID person detections
            is_two_person = len(valid_detections) == 2
            
            # Check if there's a scene change
            is_scene_change = i in scene_change_set
            
            if is_two_person and not is_scene_change:
                # Start a new segment or continue current one
                if current_segment is None:
                    current_segment = {
                        'start_idx': i,
                        'detections': []
                    }
                
                # Add detections to current segment
                current_segment['detections'].append(valid_detections)
            else:
                # End current segment if it exists
                if current_segment is not None:
                    current_segment['end_idx'] = i - 1
                    current_segment['length'] = current_segment['end_idx'] - current_segment['start_idx'] + 1
                    
                    # Only keep segments that meet minimum length requirement
                    if current_segment['length'] >= min_segment_length:
                        segments.append(current_segment)
                    
                    current_segment = None
        
        # Handle the last segment if it exists
        if current_segment is not None:
            current_segment['end_idx'] = len(frames) - 1
            current_segment['length'] = current_segment['end_idx'] - current_segment['start_idx'] + 1
            
            if current_segment['length'] >= min_segment_length:
                segments.append(current_segment)
        
        logger.info(f"Found {len(segments)} two-person segments with size filtering")
        for i, segment in enumerate(segments):
            logger.debug(f"Segment {i+1}: frames {segment['start_idx']}-{segment['end_idx']} ({segment['length']} frames)")
        
        return segments
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes on a frame for visualization.
        
        Args:
            frame: Video frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with bounding boxes drawn
        """
        viz_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        
        # Get size thresholds from config
        min_bbox_area = self.config.get('min_bbox_area', 10000)
        min_bbox_ratio = self.config.get('min_bbox_ratio', 0.01)
        min_person_height = self.config.get('min_person_height', 0.2)
        min_frame_area_threshold = frame_area * min_bbox_ratio
        min_height_threshold = frame_height * min_person_height
        
        # Get maximum thresholds
        max_bbox_ratio = self.config.get('max_bbox_ratio', 0.4)
        max_person_height = self.config.get('max_person_height', 0.9)
        max_edge_proximity = self.config.get('max_edge_proximity', 0.05)
        aspect_ratio_range = self.config.get('aspect_ratio_range', [0.3, 0.9])
        
        max_frame_area_threshold = frame_area * max_bbox_ratio
        max_height_threshold = frame_height * max_person_height
        max_edge_dist_x = int(frame_width * max_edge_proximity)
        max_edge_dist_y = int(frame_height * max_edge_proximity)
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Calculate detection metrics
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            area_ratio = area / frame_area
            height_ratio = height / frame_height
            aspect_ratio = width / height if height > 0 else 0
            
            # Calculate edge distances
            left_edge_dist = bbox[0]
            right_edge_dist = frame_width - bbox[2]
            top_edge_dist = bbox[1]
            bottom_edge_dist = frame_height - bbox[3]
            
            # Check if detection meets size requirements
            too_small = (area < min_bbox_area or 
                        area < min_frame_area_threshold or 
                        height < min_height_threshold)
                        
            too_large = (area > max_frame_area_threshold or 
                        height > max_height_threshold)
                        
            bad_aspect_ratio = (aspect_ratio < aspect_ratio_range[0] or 
                              aspect_ratio > aspect_ratio_range[1])
                              
            too_close_to_edge = (left_edge_dist < max_edge_dist_x or 
                               right_edge_dist < max_edge_dist_x or
                               top_edge_dist < max_edge_dist_y or
                               bottom_edge_dist < max_edge_dist_y)
            
            is_valid_size = not (too_small or too_large or bad_aspect_ratio or too_close_to_edge)
            
            # Choose color based on validity (green for valid, red for invalid)
            color = (0, 255, 0) if is_valid_size else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(viz_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Draw confidence and size info
            info_text = f"Conf: {confidence:.2f}"
            cv2.putText(viz_frame, 
                       info_text, 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
            size_text = f"Area: {area/1000:.1f}K ({area_ratio*100:.1f}%)"
            cv2.putText(viz_frame, 
                       size_text, 
                       (bbox[0], bbox[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
            height_text = f"Height: {height}px ({height_ratio*100:.1f}%)"
            cv2.putText(viz_frame, 
                       height_text, 
                       (bbox[0], bbox[1] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
            ratio_text = f"W/H: {aspect_ratio:.2f}"
            cv2.putText(viz_frame, 
                       ratio_text, 
                       (bbox[0], bbox[1] - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # If invalid, display reason
            if not is_valid_size:
                reason = ""
                if too_small:
                    reason = "Too small"
                elif too_large:
                    reason = "Too large/close to camera"
                elif bad_aspect_ratio:
                    reason = "Bad aspect ratio"
                elif too_close_to_edge:
                    reason = "Partially outside frame"
                    
                cv2.putText(viz_frame, 
                           reason, 
                           (bbox[0], bbox[1] - 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return viz_frame 