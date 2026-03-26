"""Quick test to visualize model predictions"""
import torch
import matplotlib.pyplot as plt
from src.models.depth_model import DepthEstimationModel
from src.data.kitti_dataset import get_kitti_loaders

def main():
    # Load model
    print("Loading model...")
    model = DepthEstimationModel(pretrained=False)
    checkpoint = torch.load('outputs/checkpoints/best.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Load data (num_workers=0 for macOS compatibility)
    print("\nLoading validation data...")
    _, val_loader = get_kitti_loaders('data/kitti', batch_size=1, num_workers=0)

    # Get one sample
    print("Getting sample...")
    batch = next(iter(val_loader))
    image = batch['image']
    depth_gt = batch['depth']

    # Predict
    print("Running prediction...")
    with torch.no_grad():
        depth_pred = model(image)

    # Visualize
    print("Creating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image[0].permute(1, 2, 0))
    axes[0].set_title('Input RGB Image', fontsize=14)
    axes[0].axis('off')

    im1 = axes[1].imshow(depth_gt[0, 0], cmap='plasma', vmin=0, vmax=80)
    axes[1].set_title('Ground Truth Depth', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Depth (m)')

    im2 = axes[2].imshow(depth_pred[0, 0], cmap='plasma', vmin=0, vmax=80)
    axes[2].set_title('Predicted Depth', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Depth (m)')

    plt.tight_layout()
    plt.savefig('outputs/sample_prediction.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to outputs/sample_prediction.png")
    plt.close()

    # Print stats
    print(f"\nModel Stats:")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    print(f"   Current Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Compute error for this sample
    valid_mask = (depth_gt > 0)
    if valid_mask.sum() > 0:
        error = torch.abs(depth_pred - depth_gt) * valid_mask
        mean_error = error.sum() / valid_mask.sum()
        print(f"\nThis Sample:")
        print(f"   Mean Absolute Error: {mean_error:.2f} meters")

if __name__ == '__main__':
    main()
