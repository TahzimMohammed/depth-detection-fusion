# Monocular Depth Estimation with Object Detection Fusion

MEng Computer Science Dissertation Project

## Overview
This project implements monocular depth estimation using ResNet18 encoder-decoder architecture, enhanced with object detection-guided boundary refinement for autonomous navigation.

## Architecture
- **Encoder:** ResNet18 (pretrained on ImageNet)
- **Decoder:** U-Net style with skip connections
- **Detection:** YOLOv8 (for boundary guidance)
- **Dataset:** KITTI Eigen split

## Setup
```bash
conda create -n depth python=3.10
conda activate depth
pip install -r requirements.txt
```

## Usage
```bash
# Train baseline model
python src/train.py --config configs/baseline.yaml

# Evaluate
python src/evaluate.py --checkpoint outputs/checkpoints/best.pth
```

## Results
- Baseline AbsRel: TBD
- With Fusion AbsRel: TBD

## References
- MonoDepth2 (Godard et al., ICCV 2019)
- SharpNet (Ramamonjisoa & Lepetit, ICCV 2019)
- YOLOv8 (Ultralytics, 2023)

