# Object Detection-Guided Monocular Depth Estimation: A Study of
Fusion Effectiveness

MEng Dissertation Project - University of Leeds, 2024-2025

## Overview

Investigates whether object detection (YOLOv8) can guide depth refinement for monocular depth estimation at reduced computational cost compared to semantic segmentation methods.

## Key Findings

- **Baseline**: Abs Rel 0.315 achieved through systematic hyperparameter optimization
- **Fusion Result**: No measurable improvement (p=0.92, Cohen's d=0.008)
- **Effectiveness Threshold**: Fusion works below Abs Rel ≈0.15, fails above it
- **Conclusion**: Semantic fusion requires high-quality baselines; low-quality baselines need architectural improvements first

## Requirements
Python 3.11
PyTorch 2.0
YOLOv8 (Ultralytics)
NumPy, OpenCV, SciPy, Matplotlib, Pillow
KITTI Dataset

## Usage

**Training:**
```bash
python train_optimized_full.py
```

**Evaluation with Fusion:**
```bash
python evaluate_optimized_with_fusion_correct.py
```

**Statistical Analysis:**
```bash
python statistical_analysis.py
```

## Results

| Method | Abs Rel | RMSE | δ<1.25 |
|--------|---------|------|--------|
| Baseline | 0.315 | 13.17m | 78.3% |
| Fusion | 0.315 | 13.17m | 78.3% |

**Literature Comparison:**
- SharpNet (baseline 0.119): +16.0% improvement
- Kuznietsov (baseline 0.135): +5.9% improvement  
- SGDepth (baseline 0.142): +7.7% improvement
- **This work (baseline 0.315): 0.0% improvement**

## Key Contributions

1. Empirically validates effectiveness threshold pattern across published methods
2. Demonstrates proper baseline optimization before fusion evaluation
3. Provides rigorous statistical analysis of negative results
4. Offers practical guidance: pursue fusion only when Abs Rel < 0.15

## Acknowledgments

- KITTI Vision Benchmark for dataset
- Ultralytics for YOLOv8
- Monodepth2 (Godard et al., 2019) for self-supervised depth estimation framework
