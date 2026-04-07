"""
Boundary-Specific Evaluation Metrics
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.ndimage import sobel


def compute_boundary_mask(depth_map: np.ndarray, 
                         boundary_width: int = 5) -> np.ndarray:
    """
    Create a mask of pixels near depth discontinuities.
    """
    # Compute depth gradients
    grad_x = sobel(depth_map, axis=1)
    grad_y = sobel(depth_map, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold to find edges
    threshold = np.percentile(gradient_magnitude, 75)
    edge_mask = gradient_magnitude > threshold
    
    # Dilate to create boundary region
    from scipy.ndimage import binary_dilation
    structure = np.ones((boundary_width, boundary_width))
    boundary_mask = binary_dilation(edge_mask, structure=structure)
    
    return boundary_mask.astype(np.uint8)


def compute_boundary_error(pred_depth: np.ndarray,
                           gt_depth: np.ndarray,
                           boundary_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute depth error specifically at boundaries.
    """
    # Extract boundary pixels only
    boundary_pred = pred_depth[boundary_mask > 0]
    boundary_gt = gt_depth[boundary_mask > 0]
    
    # Remove invalid depths
    valid = (boundary_gt > 0) & np.isfinite(boundary_gt) & np.isfinite(boundary_pred)
    boundary_pred = boundary_pred[valid]
    boundary_gt = boundary_gt[valid]
    
    if len(boundary_gt) == 0:
        return {'boundary_abs_rel': 0.0, 'boundary_rmse': 0.0}
    
    # Compute metrics
    abs_rel = np.mean(np.abs(boundary_pred - boundary_gt) / boundary_gt)
    rmse = np.sqrt(np.mean((boundary_pred - boundary_gt) ** 2))
    
    return {
        'boundary_abs_rel': float(abs_rel),
        'boundary_rmse': float(rmse),
        'boundary_pixels': int(len(boundary_gt))
    }


def compare_baseline_vs_fusion(baseline_depth: np.ndarray,
                               refined_depth: np.ndarray,
                               gt_depth: np.ndarray,
                               detections: List[Dict]) -> Dict:
    """
    Compare baseline and fusion performance at boundaries.
    """
    # Compute boundary mask
    boundary_mask = compute_boundary_mask(gt_depth, boundary_width=5)
    
    # Metrics for baseline
    baseline_metrics = compute_boundary_error(baseline_depth, gt_depth, boundary_mask)
    
    # Metrics for fusion
    fusion_metrics = compute_boundary_error(refined_depth, gt_depth, boundary_mask)
    
    # Compute improvement
    improvement = {
        'abs_rel_improvement': baseline_metrics['boundary_abs_rel'] - fusion_metrics['boundary_abs_rel'],
        'rmse_improvement': baseline_metrics['boundary_rmse'] - fusion_metrics['boundary_rmse'],
        'abs_rel_percent_change': ((baseline_metrics['boundary_abs_rel'] - fusion_metrics['boundary_abs_rel']) / 
                                   baseline_metrics['boundary_abs_rel'] * 100) if baseline_metrics['boundary_abs_rel'] > 0 else 0
    }
    
    return {
        'baseline': baseline_metrics,
        'fusion': fusion_metrics,
        'improvement': improvement,
        'num_detections': len(detections)
    }


# Test
if __name__ == '__main__':
    print("Testing boundary metrics...")
    
    # Create test data
    depth_gt = np.random.rand(192, 640) * 50 + 10
    depth_baseline = depth_gt + np.random.randn(192, 640) * 2
    depth_refined = depth_baseline.copy()
    
    # Simulate fusion improvement in a region
    depth_refined[50:100, 100:200] = depth_gt[50:100, 100:200] + np.random.randn(50, 100) * 0.5
    
    detections = [{'bbox': (100, 50, 200, 100)}]
    
    results = compare_baseline_vs_fusion(depth_baseline, depth_refined, depth_gt, detections)
    
    print(f"\nBaseline boundary error: {results['baseline']['boundary_abs_rel']:.4f}")
    print(f"Fusion boundary error: {results['fusion']['boundary_abs_rel']:.4f}")
    print(f"Improvement: {results['improvement']['abs_rel_percent_change']:.1f}%")
    print("\nBoundary metrics working!")
