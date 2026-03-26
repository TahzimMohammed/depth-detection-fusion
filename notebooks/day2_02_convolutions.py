"""
Exercise 2: Understanding Convolutions
How CNNs extract features from images
"""
import torch
import torch.nn as nn

print("="*60)
print("EXERCISE 2: CONVOLUTIONS")
print("="*60)

# Input: batch of RGB images
input_images = torch.randn(4, 3, 192, 640)
print(f"\nInput: {input_images.shape}")
print(f"  4 images, 3 channels (RGB), 192x640 pixels")

# A convolution layer
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # Extract 64 features
    kernel_size=3,      # 3x3 filter
    stride=1,           # Move 1 pixel at a time
    padding=1           # Keep same size
)

print(f"\nConvolution layer:")
print(f"  Input channels: 3 (RGB)")
print(f"  Output channels: 64 (64 different features)")
print(f"  Kernel size: 3x3")

# Apply convolution
output = conv(input_images)

print(f"\nOutput: {output.shape}")
print(f"  4 images, 64 feature maps, 192x640 pixels")

print(f"\n💡 What happened:")
print(f"  → Scanned 3x3 filters across the image")
print(f"  → Extracted 64 different features")
print(f"  → Each feature detects different patterns:")
print(f"      • Feature 1: horizontal edges")
print(f"      • Feature 2: vertical edges")
print(f"      • Feature 3: corners")
print(f"      • ... (64 total)")

# With stride=2 (downsampling)
conv_downsample = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
output_small = conv_downsample(input_images)

print(f"\nWith stride=2 (downsampling):")
print(f"  Output: {output_small.shape}")
print(f"  → Image size halved: 192→96, 640→320")
print(f"  → This is how encoders reduce spatial size!")

# Moving to GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
conv = conv.to(device)
input_images = input_images.to(device)
output_gpu = conv(input_images)

print(f"\nConvolution on GPU: {output_gpu.device}")

print("\n" + "="*60)
print("Exercise 2 Complete!")
print("You now understand how CNNs extract features!")
print("="*60)
