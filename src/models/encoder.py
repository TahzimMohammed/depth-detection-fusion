"""
ResNet18 Encoder for Depth Estimation
Extracts multi-scale features from input images
"""
import torch
import torch.nn as nn
from torchvision import models


class ResNet18Encoder(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # First layers (stem)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        print("ResNet18 Encoder initialized")
        if pretrained:
            print("Using ImageNet pretrained weights")
    
    def forward(self, x):
        x = self.conv1(x)       # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, H/4, W/4]
        
        # Encoder layers
        feat1 = self.layer1(x) # [B, 64, H/4, W/4]
        feat2 = self.layer2(feat1)  # [B, 128, H/8, W/8]
        feat3 = self.layer3(feat2)  # [B, 256, H/16, W/16]
        feat4 = self.layer4(feat3)  # [B, 512, H/32, W/32]
        
        return [feat1, feat2, feat3, feat4]


if __name__ == "__main__":
    print("="*60)
    print("Testing ResNet18 Encoder")
    print("="*60)
    
    # Create encoder
    encoder = ResNet18Encoder(pretrained=True)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 192, 640)
    print(f"\nInput: {dummy_input.shape}")
    
    # Forward pass
    features = encoder(dummy_input)
    
    print(f"\nExtracted features at 4 scales:")
    for i, feat in enumerate(features, 1):
        print(f"  Scale {i}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nParameters:")
    print(f" Total: {total_params:,}")
    print(f" Trainable: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("Encoder working perfectly!")
    print("="*60)
