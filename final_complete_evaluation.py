"""
FINAL COMPLETE EVALUATION 
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
import matplotlib.pyplot as plt

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
    """Find matching RGB image for depth GT."""
    depth_path = Path(depth_gt_path)
    parts = depth_path.parts
    
    drive_name = None
    for part in parts:
        if 'drive' in part and 'sync' in part:
            drive_name = part
            break
    
    if not drive_name:
        return None
    
    date = '_'.join(drive_name.split('_')[:3])
    camera = depth_path.parent.name
    frame = depth_path.stem
    
    raw_dir = Path(raw_data_dir)
    rgb_path = raw_dir / date / drive_name / camera / 'data' / f"{frame}.png"
    
    if rgb_path.exists():
        return str(rgb_path)
    
    alt_camera = 'image_02' if camera == 'image_03' else 'image_03'
    rgb_path_alt = raw_dir / date / drive_name / alt_camera / 'data' / f"{frame}.png"
    
    if rgb_path_alt.exists():
        return str(rgb_path_alt)
    
    return None


def find_image_gt_pairs(depth_annotated_dir, raw_data_dir, max_pairs=20):
    """Match RGB images with ground truth depth."""
    depth_dir = Path(depth_annotated_dir)
    pairs = []
    
    print("Searching for ground truth files...")
    gt_files = list(depth_dir.rglob('proj_depth/groundtruth/**/*.png'))
    print(f"Found {len(gt_files)} total ground truth files")
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
    """Compute standard depth metrics."""
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


def evaluate_complete(model, detector, fusion, image_gt_pairs, device):
    """Run complete evaluation."""
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
        'per_image': [],
        'detection_stats': {
            'total_detections': 0,
            'images_with_detections': 0,
            'detections_per_image': [],
            'object_classes': {}
        }
    }
    
    model.eval()
    
    print("\nEvaluating images...")
    
    with torch.no_grad():
        for idx, (img_path, depth_gt_path) in enumerate(tqdm(image_gt_pairs, desc="Processing")):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                original_size = image.size
                
                # Load ground truth
                depth_gt_full = load_kitti_depth_gt(depth_gt_path)
                depth_gt = np.array(Image.fromarray(depth_gt_full).resize(
                    (640, 192), Image.NEAREST
                ))
                
                # Predict baseline
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
                
                # Detect objects
                detections = detector.detect(image)
                
                # Track detection stats
                results['detection_stats']['total_detections'] += len(detections)
                results['detection_stats']['detections_per_image'].append(len(detections))
                
                if len(detections) > 0:
                    results['detection_stats']['images_with_detections'] += 1
                
                for det in detections:
                    class_name = det['class_name']
                    results['detection_stats']['object_classes'][class_name] = \
                        results['detection_stats']['object_classes'].get(class_name, 0) + 1
                
                # Scale detections
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
                
                # Apply fusion
                depth_refined = fusion.fuse(depth_baseline, scaled_detections, 
                                           preserve_boundaries=True)
                
                # Compute metrics
                baseline_metrics = compute_depth_metrics(depth_baseline, depth_gt)
                fusion_metrics = compute_depth_metrics(depth_refined, depth_gt)
                
                if baseline_metrics is None or fusion_metrics is None:
                    continue
                
                results['baseline_metrics'].append(baseline_metrics)
                results['fusion_metrics'].append(fusion_metrics)
                
                # Per-image results
                improvement = baseline_metrics['abs_rel'] - fusion_metrics['abs_rel']
                
                results['per_image'].append({
                    'image_id': idx + 1,
                    'image_path': str(Path(img_path).name),
                    'num_detections': len(scaled_detections),
                    'baseline_abs_rel': baseline_metrics['abs_rel'],
                    'fusion_abs_rel': fusion_metrics['abs_rel'],
                    'improvement': improvement,
                    'improved': improvement > 0
                })
                
            except Exception as e:
                print(f"\nError processing image {idx+1}: {e}")
                continue
    
    return results


def print_comprehensive_results(results):
    """Print detailed results."""
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS - ADAPTIVE FUSION STRATEGY")
    print("="*80)
    
    n_images = len(results['baseline_metrics'])
    print(f"\nSuccessfully evaluated: {n_images} images")
    
    # Detection Statistics
    print("\n" + "="*80)
    print("DETECTION STATISTICS")
    print("="*80)
    
    det_stats = results['detection_stats']
    print(f"Total detections:        {det_stats['total_detections']}")
    print(f"Images with detections:  {det_stats['images_with_detections']}/{n_images}")
    
    if det_stats['detections_per_image']:
        det_per_img = det_stats['detections_per_image']
        print(f"Avg detections/image:    {np.mean(det_per_img):.1f}")
        print(f"Min/Max detections:      {min(det_per_img)} / {max(det_per_img)}")
    
    print(f"\nObject classes detected:")
    for class_name, count in sorted(det_stats['object_classes'].items(), 
                                     key=lambda x: x[1], reverse=True):
        percentage = (count / det_stats['total_detections']) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Overall Metrics
    print("\n" + "="*80)
    print("OVERALL DEPTH ACCURACY")
    print("="*80)
    
    baseline_abs_rel = [m['abs_rel'] for m in results['baseline_metrics']]
    fusion_abs_rel = [m['abs_rel'] for m in results['fusion_metrics']]
    
    baseline_rmse = [m['rmse'] for m in results['baseline_metrics']]
    fusion_rmse = [m['rmse'] for m in results['fusion_metrics']]
    
    baseline_delta1 = [m['delta_1'] for m in results['baseline_metrics']]
    fusion_delta1 = [m['delta_1'] for m in results['fusion_metrics']]
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Fusion':<15} {'Change':<15}")
    print("-" * 80)
    
    abs_rel_change = (np.mean(baseline_abs_rel) - np.mean(fusion_abs_rel)) / np.mean(baseline_abs_rel) * 100
    rmse_change = (np.mean(baseline_rmse) - np.mean(fusion_rmse)) / np.mean(baseline_rmse) * 100
    delta1_change = (np.mean(fusion_delta1) - np.mean(baseline_delta1)) / np.mean(baseline_delta1) * 100
    
    print(f"{'Abs Rel Error':<20} {np.mean(baseline_abs_rel):<15.4f} {np.mean(fusion_abs_rel):<15.4f} {abs_rel_change:>13.2f}%")
    print(f"{'RMSE (m)':<20} {np.mean(baseline_rmse):<15.4f} {np.mean(fusion_rmse):<15.4f} {rmse_change:>13.2f}%")
    print(f"{'δ < 1.25':<20} {np.mean(baseline_delta1):<15.4f} {np.mean(fusion_delta1):<15.4f} {delta1_change:>13.2f}%")
    
    # Success Rate
    improved_count = sum(1 for img in results['per_image'] if img['improved'])
    success_rate = (improved_count / len(results['per_image'])) * 100
    
    print(f"\n{'Images improved:':<20} {improved_count} / {len(results['per_image'])} ({success_rate:.1f}%)")
    
    # Per-Image Breakdown
    print("\n" + "="*80)
    print("PER-IMAGE BREAKDOWN")
    print("="*80)
    
    print(f"\n{'ID':<5} {'Detections':<12} {'Baseline':<12} {'Fusion':<12} {'Change':<12} {'Status':<10}")
    print("-" * 80)
    
    for img in results['per_image']:
        status = "Better" if img['improved'] else "Worse"
        print(f"{img['image_id']:<5} {img['num_detections']:<12} "
              f"{img['baseline_abs_rel']:<12.4f} {img['fusion_abs_rel']:<12.4f} "
              f"{img['improvement']:<12.6f} {status:<10}")
    
    print("\n" + "="*80)


def save_results(results, output_path):
    """Save results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    print("="*80)
    print("FINAL COMPLETE EVALUATION - ADAPTIVE FUSION")
    print("="*80)
    print("\nThis is the official evaluation for the dissertation.\n")
    
    # Configuration
    DEPTH_ANNOTATED_DIR = 'data/kitti/data_depth_annotated/train'
    RAW_DATA_DIR = 'data/kitti/raw'
    MODEL_PATH = 'outputs/checkpoints/latest.pth'
    MAX_IMAGES = 20
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\n1.Loading baseline depth model...")
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
    print("Model loaded successfully")
    
    # Initialize detector and fusion
    print("\n2.Initializing YOLOv8 detector and adaptive fusion...")
    detector = ObjectDetector(conf_threshold=0.5)
    fusion = DepthFusion(
        filter_size=5,
        min_box_size=10,
        boundary_width=2,
        fusion_strategy='adaptive'
    )
    print("Detector and fusion ready")
    
    # Find image pairs
    print(f"\n3.Finding image-GT pairs (target: {MAX_IMAGES} images)...")
    pairs = find_image_gt_pairs(DEPTH_ANNOTATED_DIR, RAW_DATA_DIR, max_pairs=MAX_IMAGES)
    
    if len(pairs) == 0:
        print("\nNo image-GT pairs found!")
        exit(1)
    
    print(f"Found {len(pairs)} pairs")
    
    # Run evaluation
    print(f"\n4. Running complete evaluation...")
    results = evaluate_complete(model, detector, fusion, pairs, device)
    
    # Print results
    print_comprehensive_results(results)
    
    # Save results
    save_results(results, 'outputs/evaluation/final_evaluation_results.json')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nThese are your official dissertation results.")
    print("Check: outputs/evaluation/final_evaluation_results.json")
    print("\n" + "="*80)
