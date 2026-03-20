"""
Create train/val split files from available KITTI data
"""

from pathlib import Path

print("="*60)
print("Creating KITTI Split Files")
print("="*60)

# Find all available drives
raw_path = Path("data/kitti/raw/2011_09_26")
drives = sorted(raw_path.glob("*_sync"))

print(f"\nFound {len(drives)} drives:")
for drive in drives:
    print(f"  - {drive.name}")

# Collect all frames from all drives
all_samples = []

for drive in drives:
    drive_name = drive.name
    date = "2011_09_26"
    
    # Find all images in this drive
    image_dir = drive / "image_02" / "data"
    
    if image_dir.exists():
        images = sorted(image_dir.glob("*.png"))
        
        print(f"\n{drive_name}: {len(images)} frames")
        
        for img_path in images:
            frame_id = img_path.stem  # "0000000000"
            
            # Check if corresponding depth exists
            depth_path = Path("data/kitti/data_depth_annotated/train") / drive_name / "proj_depth" / "groundtruth" / "image_02" / f"{frame_id}.png"
            
            if depth_path.exists():
                # Format: "date/drive frame_id l"
                sample = f"{date}/{drive_name} {frame_id} l"
                all_samples.append(sample)
            else:
                print(f"    Warning: No depth for frame {frame_id}")

print(f"\n{'='*60}")
print(f"Total samples with depth: {len(all_samples)}")

if len(all_samples) == 0:
    print("\nERROR: No samples found!")
    print("Checking depth annotation structure...")
    depth_base = Path("data/kitti/data_depth_annotated")
    if depth_base.exists():
        print(f"\nDepth base exists. Contents:")
        for item in depth_base.iterdir():
            print(f"  {item}")
            if item.is_dir():
                for subitem in list(item.iterdir())[:5]:
                    print(f"    {subitem.name}")
else:
    # Split into train (80%) and val (20%)
    split_idx = int(len(all_samples) * 0.8)
    
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    
    # Save split files
    splits_dir = Path("data/kitti/splits")
    splits_dir.mkdir(exist_ok=True, parents=True)
    
    with open(splits_dir / "train_files.txt", 'w') as f:
        f.write('\n'.join(train_samples))
    
    with open(splits_dir / "val_files.txt", 'w') as f:
        f.write('\n'.join(val_samples))
    
    print(f"\nSplit files created:")
    print(f"  - {splits_dir / 'train_files.txt'}")
    print(f"  - {splits_dir / 'val_files.txt'}")
    
    # Show first few samples
    print(f"\nFirst 5 train samples:")
    for sample in train_samples[:5]:
        print(f"  {sample}")
    
    print(f"\nFirst 3 val samples:")
    for sample in val_samples[:3]:
        print(f"  {sample}")

print("="*60)
