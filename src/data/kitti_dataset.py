"""
KITTI Dataset Loader
Loads RGB images and corresponding depth maps for training
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class KITTIDepthDataset(Dataset):
    
    def __init__(self, data_path, split_file, mode='train'):
        self.data_path = Path(data_path)
        self.mode = mode
        
        # Reading split files
        with open(split_file, 'r') as f:
            self.filenames = f.read().splitlines()
        print(f"[{mode.upper()}] Loaded {len(self.filenames)} samples")
        self.to_tensor = transforms.ToTensor()
        
        # Data augmentation for training
        if mode == 'train':
            self.color_aug = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        else:
            self.color_aug = None
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Parse filename
        line = self.filenames[idx].split()
        folder = line[0] 
        frame_id = line[1]  
        side = line[2]  
        
        # RGB image path
        rgb_path = self.data_path / "raw" / folder / "image_02" / "data" / f"{frame_id}.png"
        
        # Depth ground truth path
        date = folder.split('/')[0] 
        drive = folder.split('/')[1] 
        depth_path = self.data_path / "data_depth_annotated" / "train" / drive / "proj_depth" / "groundtruth" / "image_02" / f"{frame_id}.png"
        
        # Load RGB image
        image = Image.open(rgb_path).convert('RGB')
        
        # Load depth
        depth_png = np.array(Image.open(depth_path), dtype=np.float32)
        depth = depth_png / 256.0
        
        # Resize to standard size 
        image = image.resize((640, 192), Image.LANCZOS)
        depth = Image.fromarray(depth).resize((640, 192), Image.NEAREST)
        depth = np.array(depth)
        
        # Apply color augmentation (training only)
        if self.mode == 'train' and self.color_aug is not None:
            image = self.color_aug(image)
        
        # Convert to tensors
        image = self.to_tensor(image) 
        depth = torch.from_numpy(depth).unsqueeze(0) 
        
        # Create mask for valid depth values
        valid_mask = (depth > 0).float()
        
        return {
            'image': image,
            'depth': depth,
            'valid_mask': valid_mask,
            'filename': f"{folder}_{frame_id}"
        }


def get_kitti_loaders(data_path, batch_size=4, num_workers=2):

    # Create datasets
    train_dataset = KITTIDepthDataset(
        data_path=data_path,
        split_file=os.path.join(data_path, 'splits', 'train_files.txt'),
        mode='train'
    )
    
    val_dataset = KITTIDepthDataset(
        data_path=data_path,
        split_file=os.path.join(data_path, 'splits', 'val_files.txt'),
        mode='val'
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data loader
    print("="*60)
    print("Testing KITTI Data Loader")
    print("="*60)
    
    dataset = KITTIDepthDataset(
        data_path='data/kitti',
        split_file='data/kitti/splits/train_files.txt',
        mode='train'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load one sample
    sample = dataset[0]
    
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  Depth shape: {sample['depth'].shape}")
    print(f"  Depth range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}] meters")
    print(f"  Valid pixels: {sample['valid_mask'].sum().item():.0f} / {sample['valid_mask'].numel()}")
    print(f"  Filename: {sample['filename']}")
    
    # Test batch loading
    print("\n" + "="*60)
    print("Testing Batch Loading")
    print("="*60)
    
    train_loader, val_loader = get_kitti_loaders('data/kitti', batch_size=2)
    
    batch = next(iter(train_loader))
    print(f"\nBatch loaded:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Depths: {batch['depth'].shape}")
    print(f"  Valid masks: {batch['valid_mask'].shape}")
    
    print("\nData loader working perfectly!")
