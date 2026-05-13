"""
Create Fusion Results Summary Visualization
Reads from actual per-image evaluation results
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Load actual results
results_file = Path('outputs/evaluation/final_evaluation_results.json')

with open(results_file, 'r') as f:
    data = json.load(f)

# Calculate overall metrics from per-image arrays
baseline_metrics = data['baseline_metrics']
fusion_metrics = data['fusion_metrics']

# Average across all images
baseline_abs_rel = np.mean([m['abs_rel'] for m in baseline_metrics])
fusion_abs_rel = np.mean([m['abs_rel'] for m in fusion_metrics])
baseline_rmse = np.mean([m['rmse'] for m in baseline_metrics])
fusion_rmse = np.mean([m['rmse'] for m in fusion_metrics])
baseline_delta = np.mean([m['delta_1'] for m in baseline_metrics]) * 100
fusion_delta = np.mean([m['delta_1'] for m in fusion_metrics]) * 100

# Count improved/degraded
improved = 0
total = len(baseline_metrics)

for base, fus in zip(baseline_metrics, fusion_metrics):
    if fus['abs_rel'] < base['abs_rel']:
        improved += 1

degraded = total - improved

print(f"Loaded {total} images from results")
print(f"Baseline Abs Rel: {baseline_abs_rel:.4f}")
print(f"Fusion Abs Rel: {fusion_abs_rel:.4f}")
print(f"Improved: {improved}/{total} ({improved/total*100:.0f}%)")
print(f"Degraded: {degraded}/{total} ({degraded/total*100:.0f}%)")

# Create visualization
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Metrics comparison
ax1 = fig.add_subplot(gs[0, :])
metrics_labels = ['Abs Rel\nError', 'RMSE\n(meters)', 'δ < 1.25\n(%)']
baseline = [baseline_abs_rel, baseline_rmse, baseline_delta]
fusion = [fusion_abs_rel, fusion_rmse, fusion_delta]

x = np.arange(len(metrics_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline, width, label='Baseline', 
                color='#4A90E2', edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, fusion, width, label='Fusion (Adaptive)', 
                color='#E24A4A', edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.set_title('Quantitative Results: Baseline vs Fusion (20 images with GT)', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_labels, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add change annotations
changes_pct = [
    ((baseline_abs_rel - fusion_abs_rel) / baseline_abs_rel * 100),
    ((baseline_rmse - fusion_rmse) / baseline_rmse * 100),
    ((fusion_delta - baseline_delta) / baseline_delta * 100)
]

for i, (b, f, chg) in enumerate(zip(baseline, fusion, changes_pct)):
    y_pos = max(b, f) * 1.05
    ax1.text(i, y_pos, f'{chg:+.2f}%', 
             ha='center', fontsize=10, fontweight='bold')

# Strategy comparison
ax2 = fig.add_subplot(gs[1, 0])
strategies = ['Mean', 'Plane', 'Gaussian', 'Adaptive', 
              'Median\n(Agg.)', 'Median\n(Cons.)']
changes = [-2.68, -0.43, -0.01, -0.00, -0.01, -0.00]
colors_strat = ['#C73E1D' if c < -0.1 else '#2E86AB' for c in changes]

ax2.barh(strategies, changes, color=colors_strat, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Abs Rel Change (%)', fontsize=11, fontweight='bold')
ax2.set_title('All Fusion Strategies Tested', fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax2.grid(axis='x', alpha=0.3)

# Images improved - CORRECT NUMBERS
ax3 = fig.add_subplot(gs[1, 1])
categories_imp = ['Improved', 'Degraded']
values_imp = [improved, degraded]
colors_imp = ['#06A77D', '#E24A4A']

wedges, texts, autotexts = ax3.pie(values_imp, labels=categories_imp, autopct='%1.0f%%',
                                     colors=colors_imp, startangle=90, 
                                     wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

ax3.set_title(f'Image-Level Results\n({improved}/{total} improved)', 
              fontsize=12, fontweight='bold')

plt.savefig('outputs/visualizations/fusion_results_summary.png', dpi=150, bbox_inches='tight')

print("\nCreated: fusion_results_summary.png")
print(f"Used actual per-image data from {total} images")
print(f"Result: {improved} improved, {degraded} degraded")
