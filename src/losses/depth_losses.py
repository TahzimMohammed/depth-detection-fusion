"""
Loss Functions for Depth Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    
    def __init__(self, lambda_si=0.5, lambda_smooth=0.001):
        super().__init__()
        
        self.lambda_si = lambda_si
        self.lambda_smooth = lambda_smooth
        
        print(f"Depth Loss initialized")
        print(f"   L1 weight: 1.0")
        print(f"   Scale-invariant weight: {lambda_si}")
        print(f"   Smoothness weight: {lambda_smooth}")
    
    def forward(self, pred, target, image, valid_mask):
    
        # L1 Loss
        l1_loss = self.l1_loss(pred, target, valid_mask)
        
        # Scale-Invariant Loss
        si_loss = self.scale_invariant_loss(pred, target, valid_mask)
        
        # Edge-Aware Smoothness
        smooth_loss = self.edge_aware_smoothness(pred, image)
        
        # Combine losses
        total_loss = l1_loss + self.lambda_si * si_loss + self.lambda_smooth * smooth_loss
        
        # Return individual losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'l1': l1_loss.item(),
            'si': si_loss.item(),
            'smooth': smooth_loss.item()
        }
        
        return total_loss, loss_dict
    
    def l1_loss(self, pred, target, valid_mask):
        """L1 loss (mean absolute error) - only on valid pixels"""
        diff = torch.abs(pred - target)
        diff = diff * valid_mask
        loss = diff.sum() / (valid_mask.sum() + 1e-6)
        return loss
    
    def scale_invariant_loss(self, pred, target, valid_mask):
        """Scale-Invariant Loss from Eigen et al. 2014"""
        log_pred = torch.log(pred + 1e-6)
        log_target = torch.log(target + 1e-6)
        log_diff = (log_pred - log_target) * valid_mask
        n = valid_mask.sum() + 1e-6
        loss = (log_diff ** 2).sum() / n - 0.5 * (log_diff.sum() ** 2) / (n ** 2)
        return loss
    
    def edge_aware_smoothness(self, depth, image):
        """Edge-Aware Smoothness Loss from MonoDepth2"""
        grad_depth_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_depth_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        grad_image_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
        grad_image_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)
        
        grad_depth_x *= torch.exp(-grad_image_x)
        grad_depth_y *= torch.exp(-grad_image_y)
        
        loss = grad_depth_x.mean() + grad_depth_y.mean()
        return loss


if __name__ == "__main__":
    print("="*60)
    print("Testing Depth Loss Functions")
    print("="*60)
    
    # Create loss function
    criterion = DepthLoss(lambda_si=0.5, lambda_smooth=0.001)
    
    # Create dummy data
    batch_size = 2
    pred_depth = torch.rand(batch_size, 1, 192, 640, requires_grad=True) * 80
    target_depth = torch.rand(batch_size, 1, 192, 640) * 80
    image = torch.rand(batch_size, 3, 192, 640)
    valid_mask = torch.ones(batch_size, 1, 192, 640)
    
    print(f"\nTesting with dummy data:")
    print(f"Predicted depth: {pred_depth.shape}")
    print(f"Target depth: {target_depth.shape}")
    print(f"Image: {image.shape}")
    print(f"Valid mask: {valid_mask.shape}")
    
    # Compute loss
    total_loss, loss_dict = criterion(pred_depth, target_depth, image, valid_mask)
    
    print(f"\nLoss values:")
    print(f"Total loss: {loss_dict['total']:.4f}")
    print(f"L1 loss: {loss_dict['l1']:.4f}")
    print(f"Scale-invariant loss: {loss_dict['si']:.4f}")
    print(f"Smoothness loss: {loss_dict['smooth']:.4f}")
    
    # Test backward pass
    print(f"\nTesting backward pass:")
    total_loss.backward()
    print(f"Gradients computed successfully")
    
    # Test with partial valid mask
    print(f"\nTesting with partial valid mask:")
    pred_depth = torch.rand(batch_size, 1, 192, 640) * 80
    target_depth = torch.rand(batch_size, 1, 192, 640) * 80
    valid_mask = torch.rand(batch_size, 1, 192, 640) > 0.5
    valid_pixels = valid_mask.sum().item()
    total_pixels = valid_mask.numel()
    
    print(f"Valid pixels: {int(valid_pixels)} / {total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
    
    total_loss, loss_dict = criterion(pred_depth, target_depth, image, valid_mask.float())
    print(f"Total loss with partial mask: {loss_dict['total']:.4f}")
    
    print("\n" + "="*60)
    print("ALL LOSS FUNCTIONS WORKING!")
    print("Ready for training!")
    print("="*60)
