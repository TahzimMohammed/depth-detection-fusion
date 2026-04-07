"""
Enhanced Depth Fusion with Multiple Strategies
"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from typing import List, Dict, Tuple, Optional


class DepthFusion:
    """Improved depth fusion with adaptive refinement."""
    
    def __init__(self, 
                 filter_size: int = 5, 
                 min_box_size: int = 10,
                 boundary_width: int = 2,
                 fusion_strategy: str = 'adaptive'):
        if filter_size % 2 == 0:
            raise ValueError(f"filter_size must be odd, got {filter_size}")
        
        if min_box_size < 1:
            raise ValueError(f"min_box_size must be >= 1, got {min_box_size}")
        
        valid_strategies = ['median', 'adaptive', 'gaussian', 'plane', 'mean']
        if fusion_strategy not in valid_strategies:
            raise ValueError(f"fusion_strategy must be one of {valid_strategies}")
        
        self.filter_size = filter_size
        self.min_box_size = min_box_size
        self.boundary_width = boundary_width
        self.fusion_strategy = fusion_strategy
        
        print(f"Depth Fusion initialized:")
        print(f" - Strategy: {fusion_strategy}")
        print(f" - Filter size: {filter_size}x{filter_size}")
        print(f" - Min box size: {min_box_size}px")
        print(f" - Boundary preservation: {boundary_width}px")
    
    def fuse(self, 
             depth_map: np.ndarray, 
             detections: List[Dict],
             preserve_boundaries: bool = True) -> np.ndarray:
        """Refine depth map using object detections."""
        if depth_map.ndim != 2:
            raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")
        
        refined = depth_map.copy()
        num_processed = 0
        num_skipped = 0
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            height, width = depth_map.shape
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            
            box_width = x2 - x1
            box_height = y2 - y1
            
            if box_width < self.min_box_size or box_height < self.min_box_size:
                num_skipped += 1
                continue
            
            object_depth = depth_map[y1:y2, x1:x2].copy()
            
            # Apply fusion strategy
            if self.fusion_strategy == 'median':
                refined_region = self._apply_median(object_depth)
            elif self.fusion_strategy == 'adaptive':
                refined_region = self._apply_adaptive(object_depth)
            elif self.fusion_strategy == 'gaussian':
                refined_region = self._apply_gaussian(object_depth)
            elif self.fusion_strategy == 'plane':
                refined_region = self._apply_plane_fitting(object_depth)
            elif self.fusion_strategy == 'mean':
                refined_region = self._apply_mean(object_depth)
            else:
                refined_region = object_depth
            
            if preserve_boundaries and self.boundary_width > 0:
                mask = self._create_boundary_mask(box_width, box_height)
                blended = mask * refined_region + (1 - mask) * object_depth
                refined[y1:y2, x1:x2] = blended
            else:
                refined[y1:y2, x1:x2] = refined_region
            
            num_processed += 1
        
        print(f"Fusion complete: processed {num_processed} objects, "
              f"skipped {num_skipped} (too small)")
        
        return refined
    
    def _apply_median(self, region: np.ndarray) -> np.ndarray:
        """Conservative median filtering."""
        return median_filter(region, size=self.filter_size)
    
    def _apply_adaptive(self, region: np.ndarray) -> np.ndarray:
        """
        Adaptive refinement based on depth variation.
        """
        # Measure local variation
        local_std = np.std(region)
        
        # Adapt filtering strength based on noise level
        if local_std < 1.0:
            # Low noise - use light gaussian
            return gaussian_filter(region, sigma=0.5)
        elif local_std < 3.0:
            # Medium noise - use median
            return median_filter(region, size=self.filter_size)
        else:
            # High noise - use stronger filtering but preserve some detail
            median_filtered = median_filter(region, size=self.filter_size)
            # Blend 70% filtered + 30% original
            return 0.7 * median_filtered + 0.3 * region
    
    def _apply_gaussian(self, region: np.ndarray) -> np.ndarray:
        """Gaussian smoothing - preserves gradients better than median."""
        return gaussian_filter(region, sigma=1.0)
    
    def _apply_plane_fitting(self, region: np.ndarray) -> np.ndarray:
        """Fit plane to enforce planarity assumption."""
        h, w = region.shape
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        z_flat = region.flatten()
        
        valid = np.isfinite(z_flat) & (z_flat > 0)
        if valid.sum() < 3:
            return region
        
        x_valid = x_flat[valid]
        y_valid = y_flat[valid]
        z_valid = z_flat[valid]
        
        A = np.c_[x_valid, y_valid, np.ones(len(x_valid))]
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
            a, b, c = coeffs
            
            plane = a * x_coords + b * y_coords + c
            
            # Blend 60% plane + 40% original (less aggressive than before)
            alpha = 0.6
            refined = alpha * plane + (1 - alpha) * region
            
            return refined
        
        except np.linalg.LinAlgError:
            return median_filter(region, size=self.filter_size)
    
    def _apply_mean(self, region: np.ndarray) -> np.ndarray:
        """Uniform depth - most aggressive."""
        mean_depth = np.mean(region[region > 0])
        return np.full_like(region, mean_depth)
    
    def _create_boundary_mask(self, width: int, height: int) -> np.ndarray:
        """Create mask: 0 at edges, 1 in interior."""
        x = np.arange(width)
        y = np.arange(height)
        
        dist_x = np.minimum(x, width - 1 - x)
        dist_y = np.minimum(y, height - 1 - y)
        
        dist_x_2d = dist_x[np.newaxis, :]
        dist_y_2d = dist_y[:, np.newaxis]
        dist_edge = np.minimum(dist_x_2d, dist_y_2d)
        
        mask = (dist_edge >= self.boundary_width).astype(np.float32)
        
        return mask
    
    def compute_refinement_stats(self, 
                                 depth_baseline: np.ndarray,
                                 depth_refined: np.ndarray,
                                 detections: List[Dict]) -> Dict:
        diff = np.abs(depth_refined - depth_baseline)
        
        stats = {
            'mean_change': float(np.mean(diff)),
            'max_change': float(np.max(diff)),
            'num_objects': len(detections),
            'changed_pixels': float(np.mean(diff > 0.01) * 100)
        }
        
        return stats


if __name__ == '__main__':
    print("\nTesting all fusion strategies...")
    
    np.random.seed(42)
    depth_map = np.random.rand(192, 640) * 100
    
    detections = [
        {'bbox': (100, 50, 200, 150), 'class_name': 'car'},
        {'bbox': (300, 80, 400, 180), 'class_name': 'person'},
    ]
    
    strategies = ['median', 'adaptive', 'gaussian', 'plane', 'mean']
    
    for strategy in strategies:
        print(f"\n--- Testing: {strategy} ---")
        fusion = DepthFusion(filter_size=5, min_box_size=20, fusion_strategy=strategy)
        depth_refined = fusion.fuse(depth_map, detections)
        stats = fusion.compute_refinement_stats(depth_map, depth_refined, detections)
        print(f"Mean change: {stats['mean_change']:.4f}")
        print(f"Changed pixels: {stats['changed_pixels']:.1f}%")
    
    print("\nAll strategies tested!")
