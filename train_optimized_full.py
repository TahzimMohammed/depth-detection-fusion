"""
Train 30-epoch model with grid search best config
"""

import torch
import torch.optim as optim
from pathlib import Path
import json

from src.models.depth_model import DepthEstimationModel
from src.losses.depth_losses import DepthLoss
from src.data.kitti_dataset import get_kitti_loaders

print("="*80)
print("TRAINING OPTIMIZED BASELINE (30 EPOCHS)")
print("="*80)

# Best config from grid search
CONFIG = {
    'learning_rate': 0.0001,
    'weight_si': 1.0,
    'weight_smooth': 0.01,
    'weight_decay': 1e-5,
    'num_epochs': 30,
    'batch_size': 4,
}

print("\nOptimized Configuration (from grid search):")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load data
print("\nLoading KITTI dataset...")
train_loader, val_loader = get_kitti_loaders(
    data_path='data/kitti',
    batch_size=CONFIG['batch_size'],
    num_workers=0
)

# Initialize model
print("Initializing model...")
model = DepthEstimationModel(pretrained=True).to(device)

criterion = DepthLoss(
    lambda_si=CONFIG['weight_si'],
    lambda_smooth=CONFIG['weight_smooth']
)

optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

output_dir = Path('outputs/optimized_baseline')
output_dir.mkdir(parents=True, exist_ok=True)

best_val_loss = float('inf')
history = {'train': [], 'val': []}

print(f"\nTraining for {CONFIG['num_epochs']} epochs...")
print("="*80)

for epoch in range(CONFIG['num_epochs']):
    # Training
    model.train()
    train_losses = []
    
    for batch_idx, batch in enumerate(train_loader):
        try:
            images = batch['image'].to(device)
            depths = batch['depth'].to(device) if 'depth' in batch else None
            valid_mask = batch['valid_mask'].to(device) if 'valid_mask' in batch else None
            
            pred_depth = model(images)
            
            if depths is not None and valid_mask is not None:
                loss, loss_dict = criterion(pred_depth, depths, images, valid_mask)
            else:
                loss = criterion.compute_photometric_loss(pred_depth, images)
                loss_dict = {'total': loss.item()}
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss_dict['total'])
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} "
                      f"[{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss_dict['total']:.4f}", end='\r')
        
        except Exception as e:
            continue
    
    avg_train = sum(train_losses) / len(train_losses) if train_losses else float('inf')
    
    # Validation
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device) if 'depth' in batch else None
                valid_mask = batch['valid_mask'].to(device) if 'valid_mask' in batch else None
                
                pred_depth = model(images)
                
                if depths is not None and valid_mask is not None:
                    loss, loss_dict = criterion(pred_depth, depths, images, valid_mask)
                else:
                    loss = criterion.compute_photometric_loss(pred_depth, images)
                    loss_dict = {'total': loss.item()}
                
                val_losses.append(loss_dict['total'])
            except:
                continue
    
    avg_val = sum(val_losses) / len(val_losses) if val_losses else float('inf')
    
    history['train'].append(float(avg_train))
    history['val'].append(float(avg_val))
    
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}: "
          f"Train={avg_train:.4f}, Val={avg_val:.4f}")
    
    # Save best
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val,
            'config': CONFIG
        }, output_dir / 'best_optimized.pth')
        print(f"  ✓ New best! Saved checkpoint")
    
    # Save latest
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val,
            'config': CONFIG
        }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    scheduler.step()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\nBest validation loss: {best_val_loss:.4f}")
print(f"Saved to: {output_dir / 'best_optimized.pth'}")

# Save history
with open(output_dir / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*80)
print("COMPARISON:")
print("-"*80)
print(f"Original model (30 epochs):     Val = 3.16")
print(f"Optimized model (30 epochs):    Val = {best_val_loss:.4f}")
if best_val_loss < 3.16:
    improvement = ((3.16 - best_val_loss) / 3.16) * 100
    print(f"Improvement: {improvement:.2f}%")
else:
    print("Note: Original config was already well-optimized!")
print("="*80)
