"""
Person tracking module for the Dyadic Interaction Dataset Generator.

This module handles:
- Tracking persons across video frames
- Maintaining consistent identity for each person
- Handling occlusions and re-identification
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

class Track:
    """Class representing a tracked person with consistent identity."""
    
    def __init__(self, track_id: int, detection: Dict[str, Any], frame_idx: int):
        """
        Initialize a new track.
        
        Args:
            track_id: Unique identifier for this track
            detection: Initial detection dictionary with bbox and confidence
            frame_idx: Frame index where this track starts
        """
        self.id = track_id
        self.bboxes = [detection['bbox']]  # List of bounding boxes for each frame
        self.confidences = [detection['confidence']]  # List of confidence scores
        self.age = 0  # Number of frames since last detection
        self.hits = 1  # Number of frames with detections
        self.start_frame = frame_idx  # First frame index
        self.last_frame = frame_idx  # Most recent frame index
        self.is_active = True  # Whether this track is currently active
        self.color = self._generate_color()  # Consistent color for visualization
        
    def _generate_color(self) -> Tuple[int, int, int]:
        """Generate a random color for this track."""
        # Use track ID to generate a deterministic color
        np.random.seed(self.id)
        color = tuple(map(int, np.random.randint(0, 255, size=3).tolist()))
        return color
        
    def update(self, detection: Dict[str, Any], frame_idx: int):
        """
        Update track with a new detection.
        
        Args:
            detection: New detection dictionary with bbox and confidence
            frame_idx: Current frame index
        """
        self.bboxes.append(detection['bbox'])
        self.confidences.append(detection['confidence'])
        self.age = 0
        self.hits += 1
        self.last_frame = frame_idx
        self.is_active = True
        
    def predict(self):
        """
        Predict next position based on motion model.
        Currently implements a simple linear motion model.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        if len(self.bboxes) < 2:
            return self.bboxes[-1]
            
        # Simple linear motion model
        last_bbox = self.bboxes[-1]
        prev_bbox = self.bboxes[-2]
        
        # Calculate velocity
        velocity = [
            last_bbox[0] - prev_bbox[0],
            last_bbox[1] - prev_bbox[1],
            last_bbox[2] - prev_bbox[2],
            last_bbox[3] - prev_bbox[3]
        ]
        
        # Predict next position
        predicted_bbox = [
            last_bbox[0] + velocity[0],
            last_bbox[1] + velocity[1],
            last_bbox[2] + velocity[2],
            last_bbox[3] + velocity[3]
        ]
        
        return predicted_bbox
    
    def mark_missed(self):
        """Mark this track as missed in the current frame."""
        self.age += 1
    
    def is_confirmed(self, min_hits: int) -> bool:
        """
        Check if this track is considered confirmed.
        
        Args:
            min_hits: Minimum number of hits required for confirmation
            
        Returns:
            True if the track is confirmed, False otherwise
        """
        return self.hits >= min_hits
    
    def should_terminate(self, max_age: int) -> bool:
        """
        Check if this track should be terminated.
        
        Args:
            max_age: Maximum number of consecutive misses before termination
            
        Returns:
            True if the track should be terminated, False otherwise
        """
        return self.age > max_age
    
    def get_recent_bbox(self) -> List[int]:
        """Get the most recent bounding box."""
        return self.bboxes[-1]
    
    def get_last_confidence(self) -> float:
        """Get the most recent confidence score."""
        return self.confidences[-1]
        
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """
        Get trajectory as a list of center points.
        
        Returns:
            List of (x, y) center points
        """
        centers = []
        for bbox in self.bboxes:
            x_center = (bbox[0] + bbox[2]) // 2
            y_center = (bbox[1] + bbox[3]) // 2
            centers.append((x_center, y_center))
        return centers
        
    def __str__(self) -> str:
        """String representation of track."""
        status = "active" if self.is_active else "inactive"
        return f"Track {self.id}: {status}, {self.hits} detections, age {self.age}"


class PersonTracker:
    """Person tracking class to maintain consistent identities across frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the person tracker with configuration.
        
        Args:
            config: Dictionary containing tracking parameters
                - tracker: Tracking algorithm ('bytetrack', 'deepsort', etc.)
                - max_age: Maximum frames to keep a track alive without matching
                - min_hits: Minimum detections before a track is confirmed
                - iou_threshold: IOU threshold for detection-track association
                - visualization: Whether to generate visualization
        """
        self.tracker_type = config.get('tracker', 'bytetrack')
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        self.visualization = config.get('visualization', True)
        
        # Track management
        self.tracks = []
        self.next_id = 0
        
        logger.info(f"Initialized PersonTracker with {self.tracker_type}: "
                   f"max_age={self.max_age}, min_hits={self.min_hits}, "
                   f"iou_threshold={self.iou_threshold}")
                   
    def reset(self):
        """Reset the tracker state."""
        self.tracks = []
        self.next_id = 0
        
    def _iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IOU score between 0 and 1
        """
        # Determine intersection rectangle coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Calculate area of intersection
        if x2 < x1 or y2 < y1:
            return 0.0  # No intersection
            
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate Union area
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IOU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
        
    def _associate_detections_to_tracks(self, 
                                      detections: List[Dict[str, Any]],
                                      tracks: List[Track],
                                      threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with existing tracks using IoU.
        
        Args:
            detections: List of detection dictionaries
            tracks: List of active tracks
            threshold: IoU threshold for association
            
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))
            
        # Initialize cost matrix
        cost_matrix = np.zeros((len(detections), len(tracks)))
        
        # Calculate IoU for each detection-track pair
        for d_idx, detection in enumerate(detections):
            detection_bbox = detection['bbox']
            for t_idx, track in enumerate(tracks):
                # Get predicted location from track
                track_bbox = track.predict()
                # Calculate IoU and store as negative (for linear_sum_assignment)
                cost_matrix[d_idx, t_idx] = -self._iou(detection_bbox, track_bbox)
                
        # Use Hungarian algorithm to find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on threshold
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for row, col in zip(row_indices, col_indices):
            # Check if IoU is high enough
            if -cost_matrix[row, col] >= threshold:
                matches.append((row, col))
                if row in unmatched_detections:
                    unmatched_detections.remove(row)
                if col in unmatched_tracks:
                    unmatched_tracks.remove(col)
                    
        return matches, unmatched_detections, unmatched_tracks
        
    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections from a frame.
        
        Args:
            detections: List of detection dictionaries with bbox and confidence
            frame_idx: Current frame index
            
        Returns:
            List of tracked detections with track_id added
        """
        # Get active tracks
        active_tracks = [t for t in self.tracks if t.is_active]
        
        # Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            detections, active_tracks, self.iou_threshold
        )
        
        # Update matched tracks
        for d_idx, t_idx in matches:
            active_tracks[t_idx].update(detections[d_idx], frame_idx)
            
        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            new_track = Track(self.next_id, detections[d_idx], frame_idx)
            self.next_id += 1
            self.tracks.append(new_track)
            
        # Update unmatched tracks
        for t_idx in unmatched_tracks:
            active_tracks[t_idx].mark_missed()
            
        # Update track states
        for track in self.tracks:
            # Check if track should be terminated
            if track.should_terminate(self.max_age):
                track.is_active = False
            
        # Create result detections with track IDs
        tracked_detections = []
        for detection_idx, detection in enumerate(detections):
            # Find the track that matched this detection
            track_id = None
            for d_idx, t_idx in matches:
                if d_idx == detection_idx:
                    # Get the track index
                    track = active_tracks[t_idx]
                    track_id = track.id
                    break
                    
            # If no match was found, find the new track
            if track_id is None:
                for track in self.tracks:
                    if (track.start_frame == frame_idx and 
                        track.bboxes[0] == detection['bbox']):
                        track_id = track.id
                        break
                        
            # Add track_id to detection and add to result
            if track_id is not None:
                tracked_detection = detection.copy()
                tracked_detection['track_id'] = track_id
                tracked_detections.append(tracked_detection)
            else:
                # This should not happen, but add as fallback
                logger.warning(f"Detection without track_id at frame {frame_idx}")
                tracked_detection = detection.copy()
                tracked_detection['track_id'] = -1
                tracked_detections.append(tracked_detection)
                
        return tracked_detections
        
    def track_segment(self, 
                     segment: Dict[str, Any], 
                     frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Process a segment and track all persons within it.
        
        Args:
            segment: Segment dictionary with detections
            frames: List of all video frames
            
        Returns:
            Segment dictionary with added tracking information
        """
        logger.info(f"Tracking persons in segment {segment.get('start_idx', 0)}-{segment.get('end_idx', 0)}")
        
        # Reset tracker state for new segment
        self.reset()
        
        # Get segment detections
        detections_per_frame = segment['detections']
        segment_start_idx = segment['start_idx']
        
        # Process each frame in the segment
        tracked_detections = []
        
        for frame_offset, frame_detections in enumerate(detections_per_frame):
            # Current global frame index
            global_frame_idx = segment_start_idx + frame_offset
            
            # Update tracker with current detections
            tracked_frame_detections = self.update(frame_detections, global_frame_idx)
            tracked_detections.append(tracked_frame_detections)
        
        # Create updated segment with tracking info
        tracked_segment = segment.copy()
        tracked_segment['tracked_detections'] = tracked_detections
        
        # Add track statistics
        active_tracks = [t for t in self.tracks if t.is_confirmed(self.min_hits)]
        tracked_segment['tracks'] = [{
            'id': track.id,
            'start_frame': track.start_frame,
            'last_frame': track.last_frame,
            'hits': track.hits,
            'is_confirmed': track.is_confirmed(self.min_hits)
        } for track in active_tracks]
        
        logger.info(f"Tracked {len(active_tracks)} persons in segment")
        
        return tracked_segment
        
    def visualize_tracked_detections(self, 
                                    frame: np.ndarray, 
                                    tracked_detections: List[Dict[str, Any]],
                                    draw_trajectory: bool = True,
                                    trajectory_length: int = 30) -> np.ndarray:
        """
        Draw tracking results on a frame.
        
        Args:
            frame: Video frame
            tracked_detections: List of detections with track_id
            draw_trajectory: Whether to draw trajectory lines
            trajectory_length: Number of past frames to show in trajectory
            
        Returns:
            Frame with tracking visualization
        """
        viz_frame = frame.copy()
        
        # Draw each detection with its track ID
        for detection in tracked_detections:
            track_id = detection.get('track_id', -1)
            if track_id == -1:
                continue
                
            # Find corresponding track
            track = None
            for t in self.tracks:
                if t.id == track_id:
                    track = t
                    break
                    
            if not track:
                continue
                
            # Get bounding box and color
            bbox = detection['bbox']
            confidence = detection['confidence']
            color = track.color
            
            # Draw bounding box
            cv2.rectangle(viz_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Draw track ID and confidence
            text = f"ID:{track_id} ({confidence:.2f})"
            cv2.putText(viz_frame, 
                       text, 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
            # Draw trajectory if enabled
            if draw_trajectory and track.hits > 1:
                # Get trajectory points
                trajectory = track.get_trajectory()
                # Limit trajectory length
                if len(trajectory) > trajectory_length:
                    trajectory = trajectory[-trajectory_length:]
                    
                # Draw lines connecting trajectory points
                for i in range(1, len(trajectory)):
                    p1 = trajectory[i-1]
                    p2 = trajectory[i]
                    cv2.line(viz_frame, p1, p2, color, 2)
        
        return viz_frame
        
    def create_track_visualization(self, segment: Dict[str, Any], frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Create visualization frames for a tracked segment.
        
        Args:
            segment: Tracked segment dictionary
            frames: List of all video frames
            
        Returns:
            List of visualization frames
        """
        if not self.visualization:
            return []
            
        logger.info(f"Creating track visualization for segment")
        
        viz_frames = []
        tracked_detections = segment['tracked_detections']
        segment_start = segment['start_idx']
        
        for frame_offset, frame_dets in enumerate(tracked_detections):
            global_frame_idx = segment_start + frame_offset
            if 0 <= global_frame_idx < len(frames):
                frame = frames[global_frame_idx]
                viz_frame = self.visualize_tracked_detections(frame, frame_dets)
                viz_frames.append(viz_frame)
                
        return viz_frames
        
    def get_person_tracks(self, segment: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group tracked detections by person identity.
        
        Args:
            segment: Tracked segment dictionary
            
        Returns:
            Dictionary mapping track_id to list of detections for that person
        """
        person_tracks = defaultdict(list)
        
        tracked_detections = segment['tracked_detections']
        segment_start = segment['start_idx']
        
        # Group detections by track_id
        for frame_offset, frame_dets in enumerate(tracked_detections):
            global_frame_idx = segment_start + frame_offset
            
            for detection in frame_dets:
                track_id = detection.get('track_id')
                if track_id is not None and track_id >= 0:
                    # Create a copy with frame information
                    det_with_frame = detection.copy()
                    det_with_frame['frame_idx'] = global_frame_idx
                    det_with_frame['frame_offset'] = frame_offset
                    person_tracks[track_id].append(det_with_frame)
        
        return person_tracks 