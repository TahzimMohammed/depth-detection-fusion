"""
Training Script for Depth Estimation Model
"""
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.depth_model import DepthEstimationModel
from losses.depth_losses import DepthLoss
from data.kitti_dataset import get_kitti_loaders


class DepthTrainer:
    """Trainer for depth estimation model"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print("="*60)
        print("DEPTH ESTIMATION TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Batch size: {args['batch_size']}")
        print(f"Epochs: {args['num_epochs']}")
        print(f"Learning rate: {args['lr']}")
        print("="*60)
        
        # Create output directories
        self.checkpoint_dir = Path(args['output_dir']) / 'checkpoints'
        self.log_dir = Path(args['output_dir']) / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Model
        print("\nLoading model...")
        self.model = DepthEstimationModel(pretrained=True)
        self.model = self.model.to(self.device)
        
        # Loss
        self.criterion = DepthLoss(
            lambda_si=args['lambda_si'],
            lambda_smooth=args['lambda_smooth']
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args['lr'],
            weight_decay=args['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args['scheduler_step'],
            gamma=0.1
        )
        
        # Data loaders
        print("\nLoading KITTI dataset...")
        self.train_loader, self.val_loader = get_kitti_loaders(
            data_path=args['data_path'],
            batch_size=args['batch_size'],
            num_workers=args['num_workers']
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        self.best_val_loss = float('inf')
        self.start_epoch = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_losses = {'l1': 0, 'si': 0, 'smooth': 0}
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['image'].to(self.device)
            depths = batch['depth'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            # Forward pass
            pred_depth = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_depth, depths, images, valid_mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss_dict['total']
            epoch_losses['l1'] += loss_dict['l1']
            epoch_losses['si'] += loss_dict['si']
            epoch_losses['smooth'] += loss_dict['smooth']
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss_dict['total']:.4f} "
                      f"(L1: {loss_dict['l1']:.4f}, "
                      f"SI: {loss_dict['si']:.4f}, "
                      f"Smooth: {loss_dict['smooth']:.4f})")
        
        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch} Training Complete")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   Avg Loss: {avg_loss:.4f}")
        
        return avg_loss, avg_losses
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        val_loss = 0
        val_losses = {'l1': 0, 'si': 0, 'smooth': 0}
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                depths = batch['depth'].to(self.device)
                valid_mask = batch['valid_mask'].to(self.device)
                
                pred_depth = self.model(images)
                loss, loss_dict = self.criterion(pred_depth, depths, images, valid_mask)
                
                val_loss += loss_dict['total']
                val_losses['l1'] += loss_dict['l1']
                val_losses['si'] += loss_dict['si']
                val_losses['smooth'] += loss_dict['smooth']
        
        # Average
        num_batches = len(self.val_loader)
        avg_val_loss = val_loss / num_batches
        avg_val_losses = {k: v / num_batches for k, v in val_losses.items()}
        
        print(f"\nValidation Results:")
        print(f"   Avg Loss: {avg_val_loss:.4f}")
        print(f"   L1: {avg_val_losses['l1']:.4f}")
        print(f"   SI: {avg_val_losses['si']:.4f}")
        print(f"   Smooth: {avg_val_losses['smooth']:.4f}")
        
        return avg_val_loss, avg_val_losses
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for epoch in range(self.start_epoch, self.args['num_epochs']):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.args['num_epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_losses = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_losses = self.validate(epoch)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Update learning rate
            self.scheduler.step()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)
        
        self.writer.close()


if __name__ == "__main__":
    # Training configuration
    args = {
        'data_path': 'data/kitti',
        'output_dir': 'outputs',
        'batch_size': 4,
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'lambda_si': 0.5,
        'lambda_smooth': 0.001,
        'scheduler_step': 10,
        'num_workers': 0,  
    }
    
    # Create trainer
    trainer = DepthTrainer(args)
    
    # Start training
    trainer.train()
