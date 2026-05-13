"""
Grid Search for Depth Model Hyperparameter Optimization

Systematically explores hyperparameter space to find optimal configuration.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import itertools
from pathlib import Path
from datetime import datetime
import numpy as np

from src.models.depth_model import DepthEstimationModel
from src.losses.depth_losses import DepthLoss
from src.data.kitti_dataset import get_kitti_loaders

print("="*80)
print("GRID SEARCH FOR DEPTH MODEL HYPERPARAMETERS")
print("="*80)

# Grid search hyperparameters
GRID = {
    'learning_rate': [5e-5, 1e-4, 5e-4],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'lambda_si': [0.1, 0.5, 1.0],
    'lambda_smooth': [0.0001, 0.001, 0.01],
    'batch_size': [4, 8],  # 8 if memory allows
}

# Fixed parameters
FIXED = {
    'num_epochs': 20,  # Reduced from 30 for faster grid search
    'scheduler_step': 10,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'data_path': 'data/kitti',
}

# Calculate total combinations
total_combinations = np.prod([len(v) for v in GRID.values()])
print(f"\nTotal combinations: {total_combinations}")
print(f"Estimated time: ~{total_combinations * 1.5:.0f} hours ({total_combinations * 1.5 / 24:.1f} days)")

# Create output directory
output_dir = Path('outputs/grid_search')
output_dir.mkdir(parents=True, exist_ok=True)

# Results tracking
all_results = []

def train_configuration(config, config_id):
    """Train model with specific hyperparameter configuration."""
    
    print(f"\n{'='*80}")
    print(f"CONFIG {config_id}/{total_combinations}")
    print(f"{'='*80}")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    device = torch.device(FIXED['device'])
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_kitti_loaders(
        data_path=FIXED['data_path'],
        batch_size=config['batch_size'],
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = DepthEstimationModel(pretrained=True).to(device)
    
    # Loss function
    criterion = DepthLoss(
        lambda_si=config['lambda_si'],
        lambda_smooth=config['lambda_smooth']
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=FIXED['scheduler_step'],
        gamma=0.1
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'config': config
    }
    
    best_val_loss = float('inf')
    
    print(f"Training for {FIXED['num_epochs']} epochs...")
    
    for epoch in range(FIXED['num_epochs']):
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['image'].to(device)
                depths = batch['depth'].to(device) if 'depth' in batch else None
                valid_mask = batch['valid_mask'].to(device) if 'valid_mask' in batch else None
                
                # Forward pass
                pred_depth = model(images)
                
                # Compute loss
                if depths is not None and valid_mask is not None:
                    loss, loss_dict = criterion(pred_depth, depths, images, valid_mask)
                else:
                    # Self-supervised fallback
                    loss = criterion.compute_photometric_loss(pred_depth, images)
                    loss_dict = {'total': loss.item()}
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss_dict['total'])
                
                # Print progress every 20 batches
                if batch_idx % 20 == 0:
                    print(f"  Epoch {epoch+1}/{FIXED['num_epochs']} "
                          f"[{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss_dict['total']:.4f}", end='\r')
            
            except Exception as e:
                print(f"\n  Error in batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        
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
                
                except Exception as e:
                    continue
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        # Record history
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(avg_val_loss))
        
        print(f"\nEpoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        # Update best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Learning rate step
        scheduler.step()
    
    # Final results
    result = {
        'config_id': config_id,
        'config': config,
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save individual result
    result_file = output_dir / f'config_{config_id:03d}.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nConfig {config_id} complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Saved to: {result_file}")
    
    return result

# Generate all configurations
print("\nGenerating configurations...")
keys = list(GRID.keys())
values = list(GRID.values())
configurations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Generated {len(configurations)} configurations")

# Ask user confirmation
print("\n" + "="*80)
print("WARNING: This will take a long time!")
print(f"Estimated: {len(configurations) * 1.5:.0f} hours")
print("="*80)
response = input("\nContinue? (yes/no): ")

if response.lower() != 'yes':
    print("Grid search cancelled.")
    exit()

# Run grid search
print("\n" + "="*80)
print("STARTING GRID SEARCH")
print("="*80)

for i, config in enumerate(configurations, 1):
    try:
        result = train_configuration(config, i)
        all_results.append(result)
        
        # Save cumulative results
        summary_file = output_dir / 'grid_search_results.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
    except Exception as e:
        print(f"\nError in config {i}: {e}")
        print("Continuing to next configuration...")
        continue

# Analysis
print("\n" + "="*80)
print("GRID SEARCH COMPLETE!")
print("="*80)

# Sort by best validation loss
all_results.sort(key=lambda x: x['best_val_loss'])

print("\nTop 5 Configurations:")
print("-"*80)
for i, result in enumerate(all_results[:5], 1):
    print(f"\n{i}. Val Loss: {result['best_val_loss']:.4f}")
    print(f"   Config: {result['config']}")

# Save summary
best_config = all_results[0]
print(f"\nBest Configuration:")
print(f"Validation Loss: {best_config['best_val_loss']:.4f}")
print(f"Config: {best_config['config']}")

summary = {
    'total_configs': len(configurations),
    'completed': len(all_results),
    'best_config': best_config,
    'top_5': all_results[:5],
    'all_results': all_results
}

summary_file = output_dir / 'grid_search_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_file}")
print(f"\n{'='*80}")
