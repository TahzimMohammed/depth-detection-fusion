"""
Test YOLOv8 Fusion Pipeline - ENHANCED VISUALIZATION
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from src.models.depth_model import DepthEstimationModel
from src.models.detection import ObjectDetector
from src.models.fusion import DepthFusion


def find_kitti_image():
    search_paths = ['data/kitti/val', 'data/kitti/test', 'data/kitti/train', 'data/kitti']
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            images = list(path.rglob('*.png')) + list(path.rglob('*.jpg'))
            if images:
                return str(images[0])
    return None


def find_model_checkpoint():
    search_paths = [
        'checkpoints/baseline_model.pth',
        'checkpoints/best_model.pth',
        'outputs/checkpoints/latest.pth',
        'outputs/baseline_model.pth',
        'baseline_model.pth',
        'best_model.pth'
    ]
    for path in search_paths:
        if Path(path).exists():
            return path
    pth_files = list(Path('.').rglob('*.pth'))
    pth_files = [f for f in pth_files if 'yolov8' not in str(f).lower()]
    if pth_files:
        return str(pth_files[0])
    return None


def test_fusion_pipeline(image_path, model_checkpoint):
    print("="*60)
    print("Testing Depth + Detection Fusion Pipeline")
    print("="*60)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 1. Load model
    print("\n1. Loading baseline depth model...")
    depth_model = DepthEstimationModel().to(device)
    
    try:
        checkpoint = torch.load(model_checkpoint, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                depth_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                depth_model.load_state_dict(checkpoint['state_dict'])
            else:
                depth_model.load_state_dict(checkpoint)
        else:
            depth_model.load_state_dict(checkpoint)
        depth_model.eval()
        print(f"Depth model loaded")
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # 2. Initialize detector and fusion
    print("\n2. Initializing YOLOv8 detector...")
    detector = ObjectDetector(model_name='yolov8n.pt', conf_threshold=0.5)
    print("Detector initialized")
    
    print("\n3. Initializing fusion module...")
    # Use mean strategy for most visible effect
    fusion = DepthFusion(filter_size=5, min_box_size=10, boundary_width=2, fusion_strategy='mean')
    print("Fusion initialized")
    
    # 3. Load image
    print(f"\n4. Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"Original size: {original_size}")
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 4. Predict baseline depth
    print("\n5. Predicting baseline depth...")
    with torch.no_grad():
        depth_output = depth_model(image_tensor)
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
    
    print(f"Depth shape: {depth_baseline.shape}")
    print(f"Depth range: [{depth_baseline.min():.2f}, {depth_baseline.max():.2f}]")
    
    # 5. Detect objects
    print("\n6. Running object detection...")
    detections = detector.detect(image)
    print(f"Found {len(detections)} objects:")
    
    scaled_detections = []
    scale_x = 640 / original_size[0]
    scale_y = 192 / original_size[1]
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        x1_s, x2_s = int(x1 * scale_x), int(x2 * scale_x)
        y1_s, y2_s = int(y1 * scale_y), int(y2 * scale_y)
        
        scaled_det = det.copy()
        scaled_det['bbox'] = (x1_s, y1_s, x2_s, y2_s)
        scaled_det['original_bbox'] = det['bbox']
        scaled_detections.append(scaled_det)
        
        box_width = x2_s - x1_s
        box_height = y2_s - y1_s
        print(f"- {det['class_name']}: {det['confidence']:.2f} ({box_width}x{box_height}px)")
    
    # 6. Apply fusion
    print("\n7. Applying fusion...")
    depth_refined = fusion.fuse(depth_baseline, scaled_detections, preserve_boundaries=True)
    
    # 7. Compute statistics
    print("\n8. Computing refinement statistics...")
    stats = fusion.compute_refinement_stats(depth_baseline, depth_refined, scaled_detections)
    print(f"Mean depth change: {stats['mean_change']:.4f}")
    print(f"Max depth change: {stats['max_change']:.4f}")
    print(f"Pixels changed: {stats['changed_pixels']:.1f}%")
    
    # 8. ENHANCED VISUALIZATION
    print("\n9. Creating enhanced visualization...")
    fig = plt.figure(figsize=(18, 12))
    
    # Create 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Row 1: Input image and detections
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Input Image', fontsize=14, weight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    img_with_boxes = np.array(image.resize((640, 192)))
    ax2.imshow(img_with_boxes)
    if len(scaled_detections) > 0:
        for det in scaled_detections:
            x1, y1, x2, y2 = det['bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, color='red', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x1, max(0, y1-5), f"{det['class_name']}", 
                    color='red', fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax2.set_title(f'Detections ({len(detections)} objects)', fontsize=14, weight='bold')
    ax2.axis('off')
    
    # Row 2: Baseline and Refined depth
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(depth_baseline, cmap='plasma', vmin=depth_baseline.min(), vmax=depth_baseline.max())
    ax3.set_title('Baseline Depth', fontsize=14, weight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(depth_refined, cmap='plasma', vmin=depth_baseline.min(), vmax=depth_baseline.max())
    ax4.set_title('Refined Depth (with Detection)', fontsize=14, weight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # Row 3: DIFFERENCE MAPS (Key visualization!)
    ax5 = fig.add_subplot(gs[2, 0])
    diff = np.abs(depth_refined - depth_baseline)
    im5 = ax5.imshow(diff, cmap='hot', vmin=0, vmax=np.percentile(diff, 99))
    ax5.set_title('Absolute Difference Map', fontsize=14, weight='bold')
    ax5.axis('off')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046)
    cbar5.set_label('Depth change (m)', rotation=270, labelpad=15)
    
    ax6 = fig.add_subplot(gs[2, 1])
    # Highlight changed regions
    changed_mask = diff > 0.1  # Threshold for "significant" change
    overlay = np.zeros((*depth_baseline.shape, 3))
    overlay[:,:,0] = depth_baseline / depth_baseline.max()  # Red channel = baseline
    overlay[:,:,1] = depth_refined / depth_refined.max()     # Green channel = refined
    overlay[changed_mask] = [1, 1, 0]  # Yellow for changed regions
    im6 = ax6.imshow(overlay)
    ax6.set_title('Change Overlay (Yellow = Modified)', fontsize=14, weight='bold')
    ax6.axis('off')
    
    # Add detection boxes to difference map
    for det in scaled_detections:
        x1, y1, x2, y2 = det['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, color='cyan', linewidth=1, linestyle='--')
        ax5.add_patch(rect)
    
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'fusion_comparison_enhanced.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("Pipeline test complete!")
    print("="*60)
    
    return {
        'depth_baseline': depth_baseline,
        'depth_refined': depth_refined,
        'detections': scaled_detections,
        'stats': stats
    }


if __name__ == '__main__':
    print("🔍 Auto-detecting paths...\n")
    
    image_path = find_kitti_image()
    model_checkpoint = find_model_checkpoint()
    
    if image_path is None:
        print("Error: Could not find any KITTI images")
        sys.exit(1)
    
    if model_checkpoint is None:
        print("Error: Could not find trained model checkpoint")
        sys.exit(1)
    
    print(f"Found image: {image_path}")
    print(f"Found model: {model_checkpoint}\n")
    
    results = test_fusion_pipeline(image_path, model_checkpoint)
    
    if results and results['stats']['num_objects'] > 0:
        print("\nResults:")
        print(f" - Processed: {results['stats']['num_objects']} objects")
        print(f" - Mean change: {results['stats']['mean_change']:.4f}m")
        print(f" - Max change: {results['stats']['max_change']:.4f}m")
        print(f" - Pixels modified: {results['stats']['changed_pixels']:.1f}%")
        print("\nCheck outputs/visualizations/fusion_comparison_enhanced.png")
        print("The difference map (bottom-left) shows WHERE fusion is working!")
