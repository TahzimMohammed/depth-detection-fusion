"""
U-Net Decoder for Depth Estimation
Upsamples features and predicts depth map
"""

import torch
import torch.nn as nn


class DepthDecoder(nn.Module):
    """
    U-Net style decoder with skip connections.
    Takes encoder features and progressively upsamples to predict depth.
    """
    
    def __init__(self, num_ch_enc=[64, 128, 256, 512]):
        super().__init__()
        
        self.num_ch_enc = num_ch_enc
        
        # Decoder channel counts (gradually reduce)
        self.num_ch_dec = [256, 128, 64, 32, 16]
        
        # Upsample blocks
        # Each block: upsample + conv + skip connection from encoder
        
        # Block 4: [B, 512, 6, 20] → [B, 256, 12, 40]
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = self._make_conv_block(256 + 256, 256)  # 256 from upconv + 256 from skip
        
        # Block 3: [B, 256, 12, 40] → [B, 128, 24, 80]
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = self._make_conv_block(128 + 128, 128)  # 128 from upconv + 128 from skip
        
        # Block 2: [B, 128, 24, 80] → [B, 64, 48, 160]
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = self._make_conv_block(64 + 64, 64)  # 64 from upconv + 64 from skip
        
        # Block 1: [B, 64, 48, 160] → [B, 32, 96, 320]
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = self._make_conv_block(32, 32)  # No skip connection at this level
        
        # Block 0: [B, 32, 96, 320] → [B, 16, 192, 640]
        self.upconv0 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv0 = self._make_conv_block(16, 16)
        
        # Final depth prediction
        self.depth_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        print("U-Net Decoder initialized")
    
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        feat1, feat2, feat3, feat4 = features
        
        # Decode from smallest to largest
        
        # Block 4: Upsample bottleneck
        x = self.upconv4(feat4)  # [B, 256, H/16, W/16]
        x = torch.cat([x, feat3], dim=1)  # Skip connection
        x = self.conv4(x)  # [B, 256, H/16, W/16]
        
        # Block 3
        x = self.upconv3(x)  # [B, 128, H/8, W/8]
        x = torch.cat([x, feat2], dim=1)  # Skip connection
        x = self.conv3(x)  # [B, 128, H/8, W/8]
        
        # Block 2
        x = self.upconv2(x)  # [B, 64, H/4, W/4]
        x = torch.cat([x, feat1], dim=1)  # Skip connection
        x = self.conv2(x)  # [B, 64, H/4, W/4]
        
        # Block 1
        x = self.upconv1(x)  # [B, 32, H/2, W/2]
        x = self.conv1(x)  # [B, 32, H/2, W/2]
        
        # Block 0
        x = self.upconv0(x)  # [B, 16, H, W]
        x = self.conv0(x)  # [B, 16, H, W]
        
        # Final depth prediction
        depth = self.depth_conv(x)  # [B, 1, H, W]
        depth = self.sigmoid(depth)  # Values in [0, 1]
        
        # Scale to max depth (80 meters for KITTI)
        depth = depth * 80.0
        
        return depth


if __name__ == "__main__":
    print("="*60)
    print("Testing U-Net Decoder")
    print("="*60)
    
    # Create decoder
    decoder = DepthDecoder()
    
    # Create dummy encoder features
    feat1 = torch.randn(2, 64, 48, 160)
    feat2 = torch.randn(2, 128, 24, 80)
    feat3 = torch.randn(2, 256, 12, 40)
    feat4 = torch.randn(2, 512, 6, 20)
    
    features = [feat1, feat2, feat3, feat4]
    
    print(f"\nEncoder features:")
    for i, feat in enumerate(features, 1):
        print(f"  Scale {i}: {feat.shape}")
    
    # Forward pass
    depth = decoder(features)
    
    print(f"\nPredicted depth:")
    print(f"  Shape: {depth.shape}")
    print(f"  Range: [{depth.min():.2f}, {depth.max():.2f}] meters")
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nDecoder parameters: {total_params:,}")
    
    print("\n" + "="*60)
    print("Decoder working perfectly!")
    print("="*60)
