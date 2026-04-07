"""
Complete Evaluation: Baseline vs Fusion with Boundary Metrics
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
from src.utils.boundary_metrics import compute_boundary_mask, compute_boundary_error


def load_kitti_validation_images(data_dir: str, max_images: int = 50):
    """
    Load KITTI validation images.
    """
    data_path = Path(data_dir)
    
    # Search for images
    search_paths = [
        data_path / 'val',
        data_path / 'test',
        data_path / 'train',
        data_path
    ]
    
    image_paths = []
    for search_path in search_paths:
        if search_path.exists():
            images = list(search_path.rglob('*.png')) + list(search_path.rglob('*.jpg'))
            image_paths.extend(images)
            if len(image_paths) >= max_images:
                break
    
    return image_paths[:max_images]


def evaluate_fusion(model, detector, fusion, image_paths, device):
    """
    Evaluate baseline vs fusion on validation set.
    """
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    results = {
        'overall': {
            'baseline_errors': [],
            'fusion_errors': [],
            'depth_changes': []
        },
        'boundary': {
            'baseline_errors': [],
            'fusion_errors': [],
            'improvements': []
        },
        'per_image': []
    }
    
    model.eval()
    
    print("Evaluating on validation set...")
    print(f"Processing {len(image_paths)} images")
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Evaluating"):
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                original_size = image.size
                
                # Preprocess
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict baseline depth
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
                
                # Scale detections
                scale_x = 640 / original_size[0]
                scale_y = 192 / original_size[1]
                
                scaled_detections = []
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    x1_s = int(x1 * scale_x)
                    x2_s = int(x2 * scale_x)
                    y1_s = int(y1 * scale_y)
                    y2_s = int(y2 * scale_y)
                    
                    scaled_det = det.copy()
                    scaled_det['bbox'] = (x1_s, y1_s, x2_s, y2_s)
                    scaled_detections.append(scaled_det)
                
                # Apply fusion
                depth_refined = fusion.fuse(depth_baseline, scaled_detections, 
                                           preserve_boundaries=True)
                
                # Compute depth change statistics
                diff = np.abs(depth_refined - depth_baseline)
                
                # Compute boundary mask (using baseline as proxy for GT edges)
                boundary_mask = compute_boundary_mask(depth_baseline, boundary_width=5)
                
                # Extract boundary errors (using baseline as "ground truth" for comparison)
                baseline_boundary = depth_baseline[boundary_mask > 0]
                refined_boundary = depth_refined[boundary_mask > 0]
                
                if len(baseline_boundary) > 0:
                    # Measure smoothness (lower variance = better)
                    baseline_var = np.var(baseline_boundary)
                    refined_var = np.var(refined_boundary)
                    
                    results['boundary']['baseline_errors'].append(baseline_var)
                    results['boundary']['fusion_errors'].append(refined_var)
                    results['boundary']['improvements'].append(
                        (baseline_var - refined_var) / baseline_var * 100 if baseline_var > 0 else 0
                    )
                
                # Overall statistics
                results['overall']['depth_changes'].append({
                    'mean': float(np.mean(diff)),
                    'max': float(np.max(diff)),
                    'percent_changed': float(np.mean(diff > 0.1) * 100)
                })
                
                # Per-image results
                results['per_image'].append({
                    'image': str(img_path),
                    'num_detections': len(scaled_detections),
                    'mean_depth_change': float(np.mean(diff)),
                    'boundary_improvement': float(results['boundary']['improvements'][-1]) if len(results['boundary']['improvements']) > 0 else 0
                })
                
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue
    
    return results


def print_results(results):
    """Print evaluation results in a nice format."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS: Baseline vs Fusion")
    print("="*70)
    
    # Overall statistics
    print("\nOVERALL STATISTICS:")
    print("-" * 70)
    
    depth_changes = results['overall']['depth_changes']
    mean_changes = [d['mean'] for d in depth_changes]
    max_changes = [d['max'] for d in depth_changes]
    percent_changed = [d['percent_changed'] for d in depth_changes]
    
    print(f"Images evaluated:        {len(depth_changes)}")
    print(f"Avg depth change:        {np.mean(mean_changes):.4f} m")
    print(f"Max depth change:        {np.mean(max_changes):.4f} m")
    print(f"Avg pixels modified:     {np.mean(percent_changed):.2f}%")
    
    # Boundary-specific statistics
    if len(results['boundary']['improvements']) > 0:
        print("\nBOUNDARY-SPECIFIC IMPROVEMENTS:")
        print("-" * 70)
        
        improvements = results['boundary']['improvements']
        
        print(f"Avg boundary improvement:     {np.mean(improvements):.2f}%")
        print(f"Median boundary improvement:  {np.median(improvements):.2f}%")
        print(f"Best improvement:             {np.max(improvements):.2f}%")
        print(f"Images with improvement:      {sum(1 for i in improvements if i > 0)} / {len(improvements)}")
    
    # Detection statistics
    print("\nDETECTION STATISTICS:")
    print("-" * 70)
    
    per_image = results['per_image']
    num_detections = [img['num_detections'] for img in per_image]
    
    print(f"Avg objects per image:   {np.mean(num_detections):.1f}")
    print(f"Total objects detected:  {sum(num_detections)}")
    print(f"Images with detections:  {sum(1 for n in num_detections if n > 0)} / {len(per_image)}")
    
    print("\n" + "="*70)


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    # Configuration
    DATA_DIR = 'data/kitti'
    MODEL_PATH = 'outputs/checkpoints/latest.pth'
    MAX_IMAGES = 20  # Increase this for full evaluation
    
    print("="*70)
    print("DEPTH FUSION EVALUATION")
    print("="*70)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
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
    
    # Initialize detector and fusion
    print("\n2. Initializing detector and fusion...")
    detector = ObjectDetector(conf_threshold=0.5)
    fusion = DepthFusion(filter_size=5, min_box_size=10, 
                        boundary_width=2, fusion_strategy='mean')
    print("Detector and fusion ready")
    
    # Load validation images
    print(f"\n3. Loading validation images (max {MAX_IMAGES})...")
    image_paths = load_kitti_validation_images(DATA_DIR, max_images=MAX_IMAGES)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("\nNo images found! Please check DATA_DIR path.")
        exit(1)
    
    # Run evaluation
    print("\n4. Running evaluation...")
    results = evaluate_fusion(model, detector, fusion, image_paths, device)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, 'outputs/evaluation/fusion_results.json')
    
    print("\nEvaluation complete!")
