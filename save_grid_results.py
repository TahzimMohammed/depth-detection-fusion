"""
Save grid search results
"""

import json
from pathlib import Path

# Manual results from terminal output
results = [
    {'config_id': 8, 'val_loss': 3.1146, 'lr': 0.0005, 'si': 1.0, 'smooth': 0.01},
    {'config_id': 6, 'val_loss': 3.2963, 'lr': 0.0005, 'si': 0.5, 'smooth': 0.01},
    {'config_id': 7, 'val_loss': 3.3212, 'lr': 0.0005, 'si': 1.0, 'smooth': 0.001},
    {'config_id': 4, 'val_loss': 3.3769, 'lr': 0.0001, 'si': 1.0, 'smooth': 0.01},
    {'config_id': 2, 'val_loss': 3.3968, 'lr': 0.0001, 'si': 0.5, 'smooth': 0.01},
    {'config_id': 1, 'val_loss': 3.4331, 'lr': 0.0001, 'si': 0.5, 'smooth': 0.001},
    {'config_id': 3, 'val_loss': 3.4384, 'lr': 0.0001, 'si': 1.0, 'smooth': 0.001},
    {'config_id': 5, 'val_loss': 3.8803, 'lr': 0.0005, 'si': 0.5, 'smooth': 0.001},
]

best = results[0]

summary = {
    'total_configs': 8,
    'best_config': {
        'learning_rate': best['lr'],
        'weight_si': best['si'],
        'weight_smooth': best['smooth'],
        'val_loss': best['val_loss']
    },
    'all_results': results
}

output_dir = Path('outputs/quick_grid_search')
output_file = output_dir / 'grid_search_summary.json'

with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print("="*80)
print("GRID SEARCH SUMMARY SAVED")
print("="*80)
print(f"\nBest Configuration:")
print(f"Learning Rate: {best['lr']}")
print(f"SI Weight: {best['si']}")
print(f"Smooth Weight: {best['smooth']}")
print(f"Validation Loss: {best['val_loss']}")
print(f"\nSaved to: {output_file}")
print("="*80)
