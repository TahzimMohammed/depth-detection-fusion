"""
Evaluate optimized baseline with fusion - FIXED JSON serialization
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import json

from src.models.depth_model import DepthEstimationModel
from src.models.detection import ObjectDetector
from src.models.fusion import DepthFusion

print("="*80)
print("FUSION EVALUATION ON OPTIMIZED BASELINE")
print("="*80)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load optimized model
print("\nLoading optimized baseline model...")
model = DepthEstimationModel().to(device)
checkpoint = torch.load('outputs/optimized_baseline/best_optimized.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from epoch {checkpoint['epoch']} (val loss: {checkpoint['val_loss']:.4f})")

# Load detector and fusion
print("\nLoading YOLOv8 detector...")
detector = ObjectDetector('yolov8n.pt', conf_threshold=0.5)

print("\nInitializing fusion...")
fusion = DepthFusion(filter_size=5, fusion_strategy='adaptive')

# Find ground truth files in data_depth_annotated
gt_base = Path('data/kitti/data_depth_annotated/train')
gt_files = list(gt_base.rglob('proj_depth/groundtruth/image_02/*.png'))

print(f"\nFound {len(gt_files)} ground truth files")

# Get corresponding images (first 20)
test_pairs = []

for gt_file in sorted(gt_files)[:20]:
    parts = gt_file.parts
    drive_name = None
    for part in parts:
        if 'drive' in part and 'sync' in part:
            drive_name = part
            break
    
    if drive_name:
        date = '_'.join(drive_name.split('_')[:3])
        img_path = Path('data/kitti/raw') / date / drive_name / 'image_02' / 'data' / gt_file.name
        
        if img_path.exists():
            test_pairs.append((img_path, gt_file))

print(f"Matched {len(test_pairs)} image-GT pairs")

if len(test_pairs) == 0:
    print("\nERROR: No matching image-GT pairs found!")
    exit(1)

transform = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"\nEvaluating on {len(test_pairs)} test images...")
print("="*80)

baseline_metrics = []
fusion_metrics = []
improved_count = 0

for idx, (img_path, gt_path) in enumerate(test_pairs, 1):
    # Load image
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get baseline depth
    with torch.no_grad():
        depth_pred = model(img_tensor)
    
    depth_np = depth_pred.squeeze().cpu().numpy()
    
    # Detect objects
    detections = detector.detect(image)
    
    # Scale bboxes to depth map size (192x640)
    h_img, w_img = image.size[1], image.size[0]
    scale_x = 640 / w_img
    scale_y = 192 / h_img
    
    scaled_detections = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        scaled_detections.append({
            'bbox': [int(x1*scale_x), int(y1*scale_y), 
                    int(x2*scale_x), int(y2*scale_y)],
            'class_name': det['class_name'],
            'confidence': det['confidence']
        })
    
    # Apply fusion
    depth_fused = fusion.fuse(depth_np, scaled_detections)
    
    # Load ground truth
    depth_gt = np.array(Image.open(gt_path)).astype(np.float32) / 256.0
    depth_gt = np.array(Image.fromarray(depth_gt).resize((640, 192), Image.NEAREST))
    
    valid_mask = (depth_gt > 0) & (depth_gt < 80)
    
    if valid_mask.sum() > 100:
        # Baseline metrics
        abs_rel_base = np.mean(np.abs(depth_np[valid_mask] - depth_gt[valid_mask]) / depth_gt[valid_mask])
        rmse_base = np.sqrt(np.mean((depth_np[valid_mask] - depth_gt[valid_mask])**2))
        
        # Fusion metrics  
        abs_rel_fused = np.mean(np.abs(depth_fused[valid_mask] - depth_gt[valid_mask]) / depth_gt[valid_mask])
        rmse_fused = np.sqrt(np.mean((depth_fused[valid_mask] - depth_gt[valid_mask])**2))
        
        # CRITICAL FIX: Convert numpy types to Python floats
        baseline_metrics.append({
            'abs_rel': float(abs_rel_base),
            'rmse': float(rmse_base)
        })
        fusion_metrics.append({
            'abs_rel': float(abs_rel_fused),
            'rmse': float(rmse_fused)
        })
        
        improved = abs_rel_fused < abs_rel_base
        if improved:
            improved_count += 1
        
        status = "✓" if improved else "✗"
        change = abs_rel_base - abs_rel_fused
        print(f"{idx:2d}. Base: {abs_rel_base:.6f} | Fused: {abs_rel_fused:.6f} | Δ: {change:+.6f} {status}")
    else:
        print(f"{idx:2d}. Skipped (insufficient valid pixels)")

if len(baseline_metrics) == 0:
    print("\nERROR: No valid evaluations!")
    exit(1)

# Overall statistics
print("\n" + "="*80)
print("OVERALL RESULTS")
print("="*80)

avg_baseline_abs_rel = np.mean([m['abs_rel'] for m in baseline_metrics])
avg_fusion_abs_rel = np.mean([m['abs_rel'] for m in fusion_metrics])
avg_baseline_rmse = np.mean([m['rmse'] for m in baseline_metrics])
avg_fusion_rmse = np.mean([m['rmse'] for m in fusion_metrics])

print(f"\nBaseline (Optimized - Val=3.14):")
print(f"Abs Rel: {avg_baseline_abs_rel:.4f}")
print(f"RMSE: {avg_baseline_rmse:.4f}m")

print(f"\nFusion (Adaptive):")
print(f"Abs Rel: {avg_fusion_abs_rel:.4f}")
print(f"RMSE: {avg_fusion_rmse:.4f}m")

change_abs = avg_baseline_abs_rel - avg_fusion_abs_rel
change_pct = (change_abs / avg_baseline_abs_rel) * 100

print(f"\nChange:")
print(f"Absolute: {change_abs:+.6f}")
print(f"Percentage: {change_pct:+.2f}%")
print(f"Images improved: {improved_count}/{len(baseline_metrics)} ({improved_count/len(baseline_metrics)*100:.0f}%)")

# Comparison
print("\n" + "="*80)
print("FULL COMPARISON")
print("="*80)
print(f"Original baseline (Val=3.16):      Abs Rel ~0.384")
print(f"Optimized baseline (Val=3.14):     Abs Rel {avg_baseline_abs_rel:.4f}")
print(f"Fusion on optimized:               Abs Rel {avg_fusion_abs_rel:.4f}")
print()
print(f"Optimization improvement:          {((0.384-avg_baseline_abs_rel)/0.384*100):+.2f}%")
print(f"Fusion improvement:                {change_pct:+.2f}%")
print("="*80)

# Save results with EXPLICIT Python type conversion
results = {
    'model_info': {
        'checkpoint': 'outputs/optimized_baseline/best_optimized.pth',
        'val_loss': float(checkpoint['val_loss']),
        'epoch': int(checkpoint['epoch']),
        'config': {
            'learning_rate': float(checkpoint['config']['learning_rate']),
            'weight_si': float(checkpoint['config']['weight_si']),
            'weight_smooth': float(checkpoint['config']['weight_smooth']),
            'weight_decay': float(checkpoint['config']['weight_decay']),
            'num_epochs': int(checkpoint['config']['num_epochs']),
            'batch_size': int(checkpoint['config']['batch_size'])
        }
    },
    'baseline_metrics': baseline_metrics,  # Already converted above
    'fusion_metrics': fusion_metrics,      # Already converted above
    'summary': {
        'baseline_abs_rel': float(avg_baseline_abs_rel),
        'fusion_abs_rel': float(avg_fusion_abs_rel),
        'baseline_rmse': float(avg_baseline_rmse),
        'fusion_rmse': float(avg_fusion_rmse),
        'change_abs': float(change_abs),
        'change_pct': float(change_pct),
        'images_improved': int(improved_count),
        'total_images': int(len(baseline_metrics))
    }
}

output_dir = Path('outputs/optimized_evaluation')
output_dir.mkdir(parents=True, exist_ok=True)

# Save JSON - should work now!
try:
    with open(output_dir / 'fusion_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_dir / 'fusion_results.json'}")
except Exception as e:
    print(f"\nWarning: Could not save JSON: {e}")
    # Save as text backup
    with open(output_dir / 'fusion_results.txt', 'w') as f:
        f.write("FUSION EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Baseline Abs Rel: {avg_baseline_abs_rel:.4f}\n")
        f.write(f"Fusion Abs Rel: {avg_fusion_abs_rel:.4f}\n")
        f.write(f"Change: {change_pct:+.2f}%\n")
        f.write(f"Images improved: {improved_count}/{len(baseline_metrics)}\n")
    print(f"Saved text backup to: {output_dir / 'fusion_results.txt'}")

print("="*80)
