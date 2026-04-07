"""
Evaluation with KITTI Ground Truth - Matching Raw Images with Depth Annotations
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image

from src.models.depth_model import DepthEstimationModel
from src.models.detection import ObjectDetector
from src.models.fusion import DepthFusion


def load_kitti_depth_gt(depth_path):
    """Load KITTI ground truth depth."""
    depth_png = np.array(Image.open(depth_path), dtype=np.float32)
    depth_gt = depth_png / 256.0
    depth_gt[depth_gt == 0] = np.nan
    return depth_gt


def find_matching_rgb_image(depth_gt_path, raw_data_dir):
    """
    Find matching RGB image for a depth GT file.
    """
    depth_path = Path(depth_gt_path)
    
    # Extract info from depth path
    parts = depth_path.parts
    
    # Find drive name
    drive_name = None
    for part in parts:
        if 'drive' in part and 'sync' in part:
            drive_name = part
            break
    
    if not drive_name:
        return None
    
    # Extract date from drive name (e.g., '2011_09_26')
    date = '_'.join(drive_name.split('_')[:3])
    
    # Get camera (image_02 or image_03)
    camera = depth_path.parent.name
    
    # Get frame number
    frame = depth_path.stem
    
    # Construct raw image path
    raw_dir = Path(raw_data_dir)
    rgb_path = raw_dir / date / drive_name / camera / 'data' / f"{frame}.png"
    
    if rgb_path.exists():
        return str(rgb_path)
    
    # Try alternative camera
    alt_camera = 'image_02' if camera == 'image_03' else 'image_03'
    rgb_path_alt = raw_dir / date / drive_name / alt_camera / 'data' / f"{frame}.png"
    
    if rgb_path_alt.exists():
        return str(rgb_path_alt)
    
    return None


def find_image_gt_pairs(depth_annotated_dir, raw_data_dir, max_pairs=20):
    """
    Match depth GT files with RGB images from raw data.
    """
    depth_dir = Path(depth_annotated_dir)
    pairs = []
    
    print("   Searching for ground truth files...")
    
    # Find all GT depth files
    gt_files = list(depth_dir.rglob('proj_depth/groundtruth/**/*.png'))
    
    print(f"Found {len(gt_files)} ground truth files")
    print("Matching with RGB images...")
    
    matched = 0
    for gt_file in gt_files:
        rgb_path = find_matching_rgb_image(str(gt_file), raw_data_dir)
        
        if rgb_path:
            pairs.append((rgb_path, str(gt_file)))
            matched += 1
            
            if len(pairs) >= max_pairs:
                break
    
    print(f"Successfully matched {matched} RGB-depth pairs")
    
    return pairs


def compute_depth_metrics(pred, gt, min_depth=1e-3, max_depth=80):
    """Compute depth metrics."""
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & np.isfinite(pred)
    
    if valid.sum() < 100:
        return None
    
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    delta_1 = np.mean(thresh < 1.25)
    delta_2 = np.mean(thresh < 1.25**2)
    delta_3 = np.mean(thresh < 1.25**3)
    
    return {
        'abs_rel': float(abs_rel),
        'rmse': float(rmse),
        'delta_1': float(delta_1),
        'delta_2': float(delta_2),
        'delta_3': float(delta_3),
        'valid_pixels': int(valid.sum())
    }


def compute_boundary_metrics(pred, gt, boundary_width=5):
    """Compute boundary-specific metrics."""
    from scipy.ndimage import sobel, binary_dilation
    
    grad_x = sobel(gt, axis=1)
    grad_y = sobel(gt, axis=0)
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    
    valid_grad = gradient[np.isfinite(gradient)]
    if len(valid_grad) == 0:
        return None
    
    threshold = np.percentile(valid_grad, 75)
    edges = gradient > threshold
    
    boundary_mask = binary_dilation(edges, iterations=boundary_width)
    
    valid = boundary_mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred)
    
    if valid.sum() < 50:
        return None
    
    pred_boundary = pred[valid]
    gt_boundary = gt[valid]
    
    abs_rel = np.mean(np.abs(pred_boundary - gt_boundary) / gt_boundary)
    rmse = np.sqrt(np.mean((pred_boundary - gt_boundary) ** 2))
    
    return {
        'boundary_abs_rel': float(abs_rel),
        'boundary_rmse': float(rmse),
        'boundary_pixels': int(valid.sum())
    }


def evaluate_with_ground_truth(model, detector, fusion, image_gt_pairs, device):
    """Run evaluation."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    results = {
        'baseline_metrics': [],
        'fusion_metrics': [],
        'baseline_boundary': [],
        'fusion_boundary': [],
        'per_image': []
    }
    
    model.eval()
    
    with torch.no_grad():
        for img_path, depth_gt_path in tqdm(image_gt_pairs, desc="Evaluating"):
            try:
                image = Image.open(img_path).convert('RGB')
                original_size = image.size
                
                depth_gt_full = load_kitti_depth_gt(depth_gt_path)
                depth_gt = np.array(Image.fromarray(depth_gt_full).resize(
                    (640, 192), Image.NEAREST
                ))
                
                image_tensor = transform(image).unsqueeze(0).to(device)
                depth_output = model(image_tensor)
                
                if isinstance(depth_output, dict):
                    if 'depth' in depth_output:
                        depth_baseline = depth_output['depth']
                    elif ('disp', 0) in depth_output:
                        depth_baseline = depth_output[('disp', 0)]
                    else:
                        depth_baseline = list(depth_output.values())[0]
                elif isinstance(depth_output, (list, tuple)):
                    depth_baseline = depth_output[0]
                else:
                    depth_baseline = depth_output
                
                depth_baseline = depth_baseline.squeeze().cpu().numpy()
                
                detections = detector.detect(image)
                
                scale_x = 640 / original_size[0]
                scale_y = 192 / original_size[1]
                
                scaled_detections = []
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    scaled_det = det.copy()
                    scaled_det['bbox'] = (
                        int(x1 * scale_x), int(y1 * scale_y),
                        int(x2 * scale_x), int(y2 * scale_y)
                    )
                    scaled_detections.append(scaled_det)
                
                depth_refined = fusion.fuse(depth_baseline, scaled_detections)
                
                baseline_metrics = compute_depth_metrics(depth_baseline, depth_gt)
                fusion_metrics = compute_depth_metrics(depth_refined, depth_gt)
                
                if baseline_metrics is None or fusion_metrics is None:
                    continue
                
                baseline_boundary = compute_boundary_metrics(depth_baseline, depth_gt)
                fusion_boundary = compute_boundary_metrics(depth_refined, depth_gt)
                
                results['baseline_metrics'].append(baseline_metrics)
                results['fusion_metrics'].append(fusion_metrics)
                
                if baseline_boundary and fusion_boundary:
                    results['baseline_boundary'].append(baseline_boundary)
                    results['fusion_boundary'].append(fusion_boundary)
                
                results['per_image'].append({
                    'image': img_path,
                    'num_detections': len(scaled_detections),
                    'baseline': baseline_metrics,
                    'fusion': fusion_metrics,
                    'improvement_abs_rel': baseline_metrics['abs_rel'] - fusion_metrics['abs_rel'],
                    'improvement_rmse': baseline_metrics['rmse'] - fusion_metrics['rmse']
                })
                
            except Exception as e:
                print(f"\nError: {e}")
                continue
    
    return results


def print_results(results):
    """Print results."""
    print("\n" + "="*70)
    print("EVALUATION WITH GROUND TRUTH DEPTH")
    print("="*70)
    
    if len(results['baseline_metrics']) == 0:
        print("\nNo valid results!")
        return
    
    print(f"\nSuccessfully evaluated {len(results['baseline_metrics'])} images")
    
    print("\nOVERALL DEPTH ACCURACY:")
    print("-" * 70)
    
    baseline_abs_rel = [m['abs_rel'] for m in results['baseline_metrics']]
    fusion_abs_rel = [m['abs_rel'] for m in results['fusion_metrics']]
    baseline_rmse = [m['rmse'] for m in results['baseline_metrics']]
    fusion_rmse = [m['rmse'] for m in results['fusion_metrics']]
    baseline_delta1 = [m['delta_1'] for m in results['baseline_metrics']]
    fusion_delta1 = [m['delta_1'] for m in results['fusion_metrics']]
    
    print(f"{'Metric':<20} {'Baseline':<15} {'Fusion':<15} {'Improvement':<15}")
    print("-" * 70)
    
    abs_rel_imp = (np.mean(baseline_abs_rel) - np.mean(fusion_abs_rel)) / np.mean(baseline_abs_rel) * 100
    rmse_imp = (np.mean(baseline_rmse) - np.mean(fusion_rmse)) / np.mean(baseline_rmse) * 100
    delta1_imp = (np.mean(fusion_delta1) - np.mean(baseline_delta1)) / np.mean(baseline_delta1) * 100
    
    print(f"{'Abs Rel Error':<20} {np.mean(baseline_abs_rel):<15.4f} {np.mean(fusion_abs_rel):<15.4f} {abs_rel_imp:>13.2f}%")
    print(f"{'RMSE (m)':<20} {np.mean(baseline_rmse):<15.4f} {np.mean(fusion_rmse):<15.4f} {rmse_imp:>13.2f}%")
    print(f"{'δ < 1.25':<20} {np.mean(baseline_delta1):<15.4f} {np.mean(fusion_delta1):<15.4f} {delta1_imp:>13.2f}%")
    
    if len(results['baseline_boundary']) > 0:
        print("\nBOUNDARY-SPECIFIC ACCURACY:")
        print("-" * 70)
        
        baseline_b_abs_rel = [m['boundary_abs_rel'] for m in results['baseline_boundary']]
        fusion_b_abs_rel = [m['boundary_abs_rel'] for m in results['fusion_boundary']]
        baseline_b_rmse = [m['boundary_rmse'] for m in results['baseline_boundary']]
        fusion_b_rmse = [m['boundary_rmse'] for m in results['fusion_boundary']]
        
        b_abs_rel_imp = (np.mean(baseline_b_abs_rel) - np.mean(fusion_b_abs_rel)) / np.mean(baseline_b_abs_rel) * 100
        b_rmse_imp = (np.mean(baseline_b_rmse) - np.mean(fusion_b_rmse)) / np.mean(baseline_b_rmse) * 100
        
        print(f"{'Boundary Abs Rel':<20} {np.mean(baseline_b_abs_rel):<15.4f} {np.mean(fusion_b_abs_rel):<15.4f} {b_abs_rel_imp:>13.2f}%")
        print(f"{'Boundary RMSE':<20} {np.mean(baseline_b_rmse):<15.4f} {np.mean(fusion_b_rmse):<15.4f} {b_rmse_imp:>13.2f}%")
    
    improvements = [(r['baseline']['abs_rel'] - r['fusion']['abs_rel']) 
                   for r in results['per_image']]
    improved_count = sum(1 for imp in improvements if imp > 0)
    
    print(f"\n{'Images improved:':<20} {improved_count} / {len(improvements)} ({improved_count/len(improvements)*100:.1f}%)")
    print("\n" + "="*70)


def save_results(results, output_path):
    """Save results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    DEPTH_ANNOTATED_DIR = 'data/kitti/data_depth_annotated/train'
    RAW_DATA_DIR = 'data/kitti/raw'
    MODEL_PATH = 'outputs/checkpoints/latest.pth'
    MAX_PAIRS = 20
    
    print("="*70)
    print("EVALUATION WITH GROUND TRUTH")
    print("="*70)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    print("\n1. Loading depth model...")
    model = DepthEstimationModel().to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded")
    
    print("\n2. Initializing detector and fusion...")
    detector = ObjectDetector(conf_threshold=0.5)
    fusion = DepthFusion(filter_size=5, min_box_size=10, 
                        boundary_width=2, fusion_strategy='adaptive')
    print("Ready")
    
    print(f"\n3. Matching RGB images with ground truth (max {MAX_PAIRS})...")
    pairs = find_image_gt_pairs(DEPTH_ANNOTATED_DIR, RAW_DATA_DIR, max_pairs=MAX_PAIRS)
    
    if len(pairs) == 0:
        print("\nNo pairs found!")
        exit(1)
    
    print("\n4.Running evaluation...")
    results = evaluate_with_ground_truth(model, detector, fusion, pairs, device)
    
    print_results(results)
    save_results(results, 'outputs/evaluation/ground_truth_results.json')
    
    print("\nEvaluation complete! These are REAL metrics against ground truth.")
