"""
Complete Depth Estimation Model
ResNet18 Encoder + U-Net Decoder
"""

import torch
import torch.nn as nn

# Handle both direct execution and module imports
try:
    from .encoder import ResNet18Encoder
    from .decoder import DepthDecoder
except ImportError:
    from encoder import ResNet18Encoder
    from decoder import DepthDecoder


class DepthEstimationModel(nn.Module):
    """
    Complete monocular depth estimation model.
    
    Architecture:
        Input: RGB image [B, 3, H, W]
        ↓
        ResNet18 Encoder (pretrained on ImageNet)
        ↓
        Multi-scale features
        ↓
        U-Net Decoder (with skip connections)
        ↓
        Output: Depth map [B, 1, H, W]
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.decoder = DepthDecoder()
        
        print("="*60)
        print("✅ Complete Depth Model Initialized")
        print("="*60)
        
        # Count parameters
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = enc_params + dec_params
        
        print(f"  Encoder parameters: {enc_params:,}")
        print(f"  Decoder parameters: {dec_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print("="*60)
    
    def forward(self, x):
        """
        Predict depth from RGB image
        
        Args:
            x: RGB images [B, 3, H, W]
        
        Returns:
            depth: Predicted depth maps [B, 1, H, W] in meters
        """
        # Extract features
        features = self.encoder(x)
        
        # Predict depth
        depth = self.decoder(features)
        
        return depth


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("="*60)
    print("Testing Complete Depth Model")
    print("="*60)
    
    # Create model
    model = DepthEstimationModel(pretrained=True)
    
    # Test with realistic KITTI size
    print("\n🧪 Testing with KITTI-sized input:")
    
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 192, 640)
    
    print(f"  Input: {dummy_images.shape}")
    
    # Forward pass
    with torch.no_grad():
        predicted_depth = model(dummy_images)
    
    print(f"  Output: {predicted_depth.shape}")
    print(f"  Depth range: [{predicted_depth.min():.2f}, {predicted_depth.max():.2f}] meters")
    
    # Test on GPU/MPS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n🚀 Testing on {device}:")
    
    model = model.to(device)
    dummy_images = dummy_images.to(device)
    
    with torch.no_grad():
        predicted_depth = model(dummy_images)
    
    print(f"  ✅ Model runs on {device}")
    print(f"  Output device: {predicted_depth.device}")
    
    # Memory usage
    print(f"\n💾 Model size:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB (float32)")
    
    print("\n" + "="*60)
    print("✅ COMPLETE MODEL WORKING!")
    print("Ready to train on KITTI dataset!")
    print("="*60)
