"""
Complete Depth Estimation Model
ResNet18 Encoder + U-Net Decoder
"""

import torch
import torch.nn as nn
from .encoder import ResNet18Encoder
from .decoder import DepthDecoder


class DepthEstimationModel(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.decoder = DepthDecoder()
        
        print("="*60)
        print("Complete Depth Model Initialized")
        print("="*60)
        
        # Count parameters
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = enc_params + dec_params
        
        print(f"Encoder parameters: {enc_params:,}")
        print(f"Decoder parameters: {dec_params:,}")
        print(f"Total parameters: {total_params:,}")
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
    print("="*60)
    print("Testing Complete Depth Model")
    print("="*60)
    
    # Create model
    model = DepthEstimationModel(pretrained=True)
    
    # Test with realistic KITTI size
    print("\nTesting with KITTI-sized input:")
    
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 192, 640)
    
    print(f"Input: {dummy_images.shape}")
    
    # Forward pass
    with torch.no_grad():  # Don't compute gradients for testing
        predicted_depth = model(dummy_images)
    
    print(f"  Output: {predicted_depth.shape}")
    print(f"  Depth range: [{predicted_depth.min():.2f}, {predicted_depth.max():.2f}] meters")
    
    # Test on GPU/MPS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nTesting on {device}:")
    
    model = model.to(device)
    dummy_images = dummy_images.to(device)
    
    with torch.no_grad():
        predicted_depth = model(dummy_images)
    
    print(f"Model runs on {device}")
    print(f"Output device: {predicted_depth.device}")
    
    # Memory usage
    print(f"\nModel size:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024**2:.1f} MB (float32)")
    
    print("\n" + "="*60)
    print("COMPLETE MODEL WORKING!")
    print("Ready to train on KITTI dataset!")
    print("="*60)
