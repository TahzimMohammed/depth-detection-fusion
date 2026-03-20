"""
Day 2, Exercise 4: The Training Loop
How neural networks actually LEARN
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("="*60)
print("EXERCISE 4: TRAINING A MODEL")
print("="*60)

# Simple model
class TinyDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.final = nn.Conv2d(16, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.upconv1(x))
        x = self.relu(self.upconv2(x))
        x = self.sigmoid(self.final(x))
        return x

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TinyDepthNet().to(device)

# Loss function (measures how wrong predictions are)
criterion = nn.L1Loss()  # Mean absolute error

# Optimizer (updates model weights to reduce loss)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining Setup:")
print(f"  Model: TinyDepthNet")
print(f"  Loss: L1 Loss (mean absolute error)")
print(f"  Optimizer: Adam (learning rate = 0.001)")
print(f"  Device: {device}")

# Dummy training data
print("\nCreating dummy training data...")
print("  (In real training, this would be KITTI images)")

num_samples = 10
images = torch.randn(num_samples, 3, 192, 640).to(device)
target_depths = torch.rand(num_samples, 1, 192, 640).to(device)  # Random ground truth

print(f"Training samples: {num_samples}")
print(f"Image shape: {images[0].shape}")
print(f"Target depth shape: {target_depths[0].shape}")

# THE TRAINING LOOP
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

num_epochs = 5

for epoch in range(num_epochs):
    epoch_loss = 0
    
    # Process each training sample
    for i in range(num_samples):
        # Get one image and its ground truth depth
        image = images[i:i+1]           # [1, 3, 192, 640]
        target = target_depths[i:i+1]   # [1, 1, 192, 640]
        
        # 1. FORWARD PASS - Make prediction
        prediction = model(image)
        
        # 2. COMPUTE LOSS - How wrong is the prediction?
        loss = criterion(prediction, target)
        
        # 3. BACKWARD PASS - Calculate gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        
        # 4. UPDATE WEIGHTS - Improve the model
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Average loss for this epoch
    avg_loss = epoch_loss / num_samples
    
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# Test the trained model
print("\nTesting trained model:")
test_image = torch.randn(1, 3, 192, 640).to(device)
with torch.no_grad():  # Don't compute gradients for testing
    test_prediction = model(test_image)

print(f"  Input: {test_image.shape}")
print(f"  Predicted depth: {test_prediction.shape}")
print(f"  Depth range: [{test_prediction.min():.3f}, {test_prediction.max():.3f}]")

print("\nWhat just happened:")
print("  1. Forward pass: Model predicted depth")
print("  2. Loss computation: Measured prediction error")
print("  3. Backward pass: Calculated gradients")
print("  4. Weight update: Improved model parameters")
print("  5. Repeat: Model got better each epoch!")

print("\nIn real training:")
print("  • Use KITTI images (not random data)")
print("  • Train for 20+ epochs (not just 5)")
print("  • Use validation set to check progress")
print("  • Save best model checkpoint")

print("\n" + "="*60)
print("Exercise 4 Complete!")
print("You now understand how neural networks LEARN!")
print("="*60)
