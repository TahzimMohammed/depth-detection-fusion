"""
Day 2, Exercise 3: Building Your First Neural Network
A simplified version of your actual depth model
"""
import torch
import torch.nn as nn

print("="*60)
print("EXERCISE 3: BUILDING A NEURAL NETWORK")
print("="*60)

class TinyDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder (downsample)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        # Input: [B, 3, 192, 640] → Output: [B, 32, 96, 320]
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # Input: [B, 32, 96, 320] → Output: [B, 64, 48, 160]
        
        # Decoder (upsample)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Input: [B, 64, 48, 160] → Output: [B, 32, 96, 320]
        
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # Input: [B, 32, 96, 320] → Output: [B, 16, 192, 640]
        
        # Final prediction
        self.final = nn.Conv2d(16, 1, kernel_size=1)
        # Input: [B, 16, 192, 640] → Output: [B, 1, 192, 640]
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass - predict depth from image"""
        print(f"\n  Input: {x.shape}")
        
        # Encoder
        x = self.relu(self.conv1(x))
        print(f"  After conv1: {x.shape}")
        
        x = self.relu(self.conv2(x))
        print(f"  After conv2: {x.shape} (bottleneck - smallest size)")
        
        # Decoder
        x = self.relu(self.upconv1(x))
        print(f"  After upconv1: {x.shape}")
        
        x = self.relu(self.upconv2(x))
        print(f"  After upconv2: {x.shape}")
        
        # Final depth prediction
        x = self.sigmoid(self.final(x))
        print(f"  Output: {x.shape} (depth map!)")
        
        return x

# Create model
model = TinyDepthNet()

print("\nModel Architecture:")
print(f"  Encoder: Downsample image (extract features)")
print(f"  Decoder: Upsample features (predict depth)")
print(f"  Output: Depth map (same size as input)")

# Test with dummy data
dummy_image = torch.randn(2, 3, 192, 640)
print(f"\nTesting with 2 images:")

depth_prediction = model(dummy_image)

print(f"\nSuccess! Model predicted depth maps!")
print(f"  Input: RGB images {dummy_image.shape}")
print(f"  Output: Depth maps {depth_prediction.shape}")

# Move to GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
dummy_image = dummy_image.to(device)

depth_on_gpu = model(dummy_image)
print(f"\nModel runs on GPU: {depth_on_gpu.device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel has {total_params:,} parameters")

print("\n" + "="*60)
print("Exercise 3 Complete!")
print("You just built a neural network that predicts depth!")
print("Your real model will be similar but bigger and better!")
print("="*60)
