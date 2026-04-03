"""
Depth Fusion with Object Detection
This module implements the fusion mechanism that combines depth predictions
with object detection bounding boxes to refine depth maps.
"""

import numpy as np
from scipy.ndimage import median_filter
from typing import List, Dict, Tuple, Optional


class DepthFusion:
    """
    # Fuses depth predictions with object detection bounding boxes.
    """
    
    def __init__(self, 
                 filter_size: int = 5, 
                 min_box_size: int = 20,
                 boundary_width: int = 2):
  
        # Initialize the depth fusion module.
        if filter_size % 2 == 0:
            raise ValueError(f"filter_size must be odd, got {filter_size}")
        
        if min_box_size < 1:
            raise ValueError(f"min_box_size must be >= 1, got {min_box_size}")
        
        self.filter_size = filter_size
        self.min_box_size = min_box_size
        self.boundary_width = boundary_width
        
        print(f"Depth Fusion initialized:")
        print(f" - Filter size: {filter_size}x{filter_size}")
        print(f" - Min box size: {min_box_size}px")
        print(f" - Boundary preservation: {boundary_width}px")
    
    def fuse(self, 
             depth_map: np.ndarray, 
             detections: List[Dict],
             preserve_boundaries: bool = True) -> np.ndarray:
        """
        Refine depth map using object detections:
        This is the main fusion function. It processes each detected object
        and applies median filtering within the object region to remove
        noise and enforce geometric consistency.
        """
        # Validate input
        if depth_map.ndim != 2:
            raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")
        
        # Start with copy of original depth
        refined = depth_map.copy()
        
        # Track statistics
        num_processed = 0
        num_skipped = 0
        
        # Process each detected object
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Validate bounding box is within image bounds
            height, width = depth_map.shape
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            
            # Skip small boxes
            box_width = x2 - x1
            box_height = y2 - y1
            
            if box_width < self.min_box_size or box_height < self.min_box_size:
                num_skipped += 1
                continue
            
            # Extract object region depth
            object_depth = depth_map[y1:y2, x1:x2].copy()
            
            # Apply median filter to enforce consistency
            smoothed = median_filter(object_depth, size=self.filter_size)
            
            if preserve_boundaries and self.boundary_width > 0:
                # Preserve original depth at boundaries for sharpness
                mask = self._create_boundary_mask(box_width, box_height)
                blended = mask * smoothed + (1 - mask) * object_depth
                refined[y1:y2, x1:x2] = blended
            else:
                # Replace entire region with smoothed depth
                refined[y1:y2, x1:x2] = smoothed
            
            num_processed += 1
        
        print(f"Fusion complete: processed {num_processed} objects, " f"skipped {num_skipped} (too small)")
        
        return refined
    
    def _create_boundary_mask(self, width: int, height: int) -> np.ndarray:
        """
        Create a mask that is 0 at boundaries and 1 in the interior.
        This is used to preserve sharp boundaries while smoothing interiors.
        """
        # Create distance from edge maps
        x = np.arange(width)
        y = np.arange(height)
        
        # Distance from left/right edges
        dist_x = np.minimum(x, width - 1 - x)
        # Distance from top/bottom edges
        dist_y = np.minimum(y, height - 1 - y)
        
        # 2D distance from nearest edge
        dist_x_2d = dist_x[np.newaxis, :]
        dist_y_2d = dist_y[:, np.newaxis]
        dist_edge = np.minimum(dist_x_2d, dist_y_2d)
        
        # Create mask: 0 within boundary_width of edge, 1 elsewhere
        mask = (dist_edge >= self.boundary_width).astype(np.float32)
        
        return mask
    
    def compute_refinement_stats(self, 
                                 depth_baseline: np.ndarray,
                                 depth_refined: np.ndarray,
                                 detections: List[Dict]) -> Dict:
        """
        To Compute statistics about the refinement process.
        This helps to evaluate how much the fusion changed the depth map.
        """

        # Compute difference
        diff = np.abs(depth_refined - depth_baseline)
        
        # Statistics
        stats = {
            'mean_change': float(np.mean(diff)),
            'max_change': float(np.max(diff)),
            'num_objects': len(detections),
            'changed_pixels': float(np.mean(diff > 0.01) * 100)  # % changed
        }
        
        return stats


def test_fusion():
    # Test function to verify the fusion module works.
    print("\n" + "="*60)
    print("Testing Depth Fusion Module")
    print("="*60 + "\n")
    
    # Create dummy depth map (random noise for testing)
    depth_map = np.random.rand(192, 640) * 100  # KITTI size
    
    # Create dummy detections
    detections = [
        {'bbox': (100, 50, 200, 150), 'class_name': 'car', 'confidence': 0.9},
        {'bbox': (300, 80, 400, 180), 'class_name': 'person', 'confidence': 0.85},
        {'bbox': (500, 100, 550, 150), 'class_name': 'bicycle', 'confidence': 0.75}
    ]
    
    # Initialize fusion
    fusion = DepthFusion(filter_size=5, min_box_size=20)
    
    # Apply fusion
    print("\nApplying fusion...")
    depth_refined = fusion.fuse(depth_map, detections)
    
    # Compute statistics
    stats = fusion.compute_refinement_stats(depth_map, depth_refined, detections)
    
    print("\nRefinement Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Fusion module is ready!")
    print("\nTo use in your code:")
    print("from fusion import DepthFusion")
    print("fusion = DepthFusion(filter_size=5)")
    print("refined_depth = fusion.fuse(baseline_depth, detections)")
    print("\n" + "="*60)


if __name__ == '__main__':
    # Run test when this file is executed directly
    test_fusion()