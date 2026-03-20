import torch

print("="*60)
print("TENSOR SHAPES EXERCISE")
print("="*60)

batch = torch.randn(4, 3, 192, 640)
print(f"\nBatch shape: {batch.shape}")
print(f" → 4 images, 3 channels (RGB), 192x640 pixels")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\n Using: {device}")

batch_gpu = batch.to(device)
print(f"Moved to: {batch_gpu.device}")

print("\n" + "="*60)
print("Environment working! Ready to learn PyTorch!")
print("="*60)
