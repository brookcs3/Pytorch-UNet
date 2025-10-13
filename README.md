# PyTorch U-Net (Historical Archive)

⚠️ **This is an OLD and DEFUNCT project** ⚠️

This repository was originally created for the **Kaggle Carvana Image Masking Challenge** (2017), which is now closed. The original competition dataset is no longer publicly available.

## What This Was For

This implementation of U-Net was designed for **semantic segmentation** - specifically for automatically removing backgrounds from car images in the Carvana dataset. The model would take a photo of a car and output a binary mask showing which pixels were car vs. background.

**Original Competition:** https://www.kaggle.com/c/carvana-image-masking-challenge (CLOSED)

The model achieved a Dice coefficient of 0.988423 on the test set.

---

## Current Status

- ✅ Code still works
- ✅ Modified to support Apple Silicon MPS (GPU acceleration on Mac)
- ❌ Original Kaggle dataset no longer available
- ❌ Download scripts removed
- ✅ Can be used with custom data

---

## What is U-Net?

U-Net is a convolutional neural network architecture designed for **semantic segmentation**. It can be used for:

- Medical image segmentation (tumor detection, organ segmentation)
- Object masking and background removal
- Satellite imagery analysis
- Any pixel-wise classification task

**Architecture:**
- Encoder-decoder structure with skip connections
- Input: RGB images (3 channels)
- Output: Segmentation masks (pixel-wise class predictions)
- ~7.7M parameters

**Original Paper:** [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

## Installation

### Requirements
- Python 3.11+
- PyTorch 2.0+
- `uv` package manager

### Setup

```bash
# Clone or navigate to the repository
cd /path/to/Pytorch-UNet

# Install dependencies using uv pip
uv pip install -r requirements.txt

# Install PyTorch (if not already installed)
uv pip install torch torchvision
```

### Apple Silicon Notes

This repository has been modified to support **MPS (Metal Performance Shaders)** for GPU acceleration on Apple Silicon Macs. The device selection automatically prioritizes:

1. MPS (Apple Silicon GPU)
2. CUDA (NVIDIA GPU)
3. CPU

---

## Quick Test Commands

**Note:** Run these commands from the repository root directory:
```bash
cd Pytorch-UNet
```

### 1. Test Import
```bash
python3 -c "from unet import UNet; print('✅ U-Net imported successfully!')"
```

### 2. Test Model Loading
```bash
python3 -c "from unet import UNet; m = UNet(3, 2); print(f'✅ U-Net loaded: {sum(p.numel() for p in m.parameters()):,} parameters')"
```

### 3. Test Forward Pass (Input → Output)
```bash
python3 -c "
from unet import UNet
import torch

print('📥 INPUT:')
x = torch.randn(1, 3, 572, 572)
print(f'   Shape: {x.shape}')
print(f'   Min: {x.min():.3f}, Max: {x.max():.3f}')

print('\n⚙️  Processing through U-Net...')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=2).to(device)
model.eval()

with torch.no_grad():
    output = model(x.to(device))

print('\n📤 OUTPUT:')
print(f'   Shape: {output.shape}')
print(f'   Min: {output.min():.3f}, Max: {output.max():.3f}')
print(f'   Device: {output.device}')
print('\n✅ U-Net working on {device}!')
"
```

### 4. Test Training Pipeline (Forward + Backward Pass)
```bash
python3 -c "
import torch
from unet import UNet
from PIL import Image
import numpy as np

print('📂 Loading test data...')
img = np.array(Image.open('data/imgs/test_0.png'))
mask = np.array(Image.open('data/masks/test_0_mask.png'))

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=2).to(device)

img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
mask_tensor = torch.from_numpy(mask).unsqueeze(0).long().to(device) // 255

print('⚡ Forward pass...')
output = model(img_tensor)

print('📊 Computing loss...')
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, mask_tensor)
print(f'   Loss: {loss.item():.4f}')

print('⬅️  Backward pass...')
loss.backward()

print('✅ Training pipeline fully functional!')
"
```

### 5. Complete Mini Training Loop (Get a .pth file!)
```bash
python3 -c "
import torch
import numpy as np
from unet import UNet
from PIL import Image

print('🔧 Setup...')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
print(f'   Device: {device}')

print('\n📊 Creating fake training data...')
# Fake data (replace with real spectrograms + masks)
x = torch.rand(1, 3, 256, 256).to(device)       # input spectrogram
y = torch.randint(0, 2, (1, 256, 256)).to(device)  # target mask (class indices)

print('\n🏋️  Training for 10 epochs...')
# Training loop
for epoch in range(10):
    model.train()
    optimizer.zero_grad()

    output = model(x)               # forward pass
    loss = criterion(output, y)     # compute loss
    loss.backward()                 # backprop
    optimizer.step()                # update weights

    print(f'   Epoch {epoch+1}/10 - Loss: {loss.item():.4f}')

print('\n💾 Saving trained model...')
# Save trained weights
torch.save(model.state_dict(), 'unet_audio_model.pth')
print('✅ Model trained and saved as unet_audio_model.pth')
print('   You can now load it with: model.load_state_dict(torch.load(\"unet_audio_model.pth\"))')
"
```

---

## Using with Custom Data

Since the original Carvana dataset is unavailable, you can use your own segmentation data:

### Data Structure
```
data/
├── imgs/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/
    ├── image_001_mask.png
    ├── image_002_mask.png
    └── ...
```

**Requirements:**
- Images: RGB format (any size, will be scaled)
- Masks: Grayscale, same filename as image + `_mask` suffix
- Masks should be binary (0 = background, 255 = foreground) or multi-class

### Training Command
```bash
python3 train.py --epochs 10 --batch-size 2 --scale 0.5 --validation 20
```

**Arguments:**
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (reduce if out of memory)
- `--scale`: Image downscaling factor (0.5 = half resolution)
- `--validation`: Percentage of data for validation (0-100)
- `--amp`: Enable automatic mixed precision (CUDA only, not supported on MPS)

### Prediction Command
```bash
python3 predict.py -i path/to/image.jpg -o output_mask.png
```

---

## Model Architecture Details

```python
from unet import UNet

# Binary segmentation (background vs. foreground)
model = UNet(n_channels=3, n_classes=2, bilinear=False)

# Multi-class segmentation (e.g., 5 classes)
model = UNet(n_channels=3, n_classes=5, bilinear=False)

# With bilinear upsampling (fewer parameters, faster)
model = UNet(n_channels=3, n_classes=2, bilinear=True)
```

**Parameters:**
- `n_channels`: Number of input channels (3 for RGB, 1 for grayscale)
- `n_classes`: Number of output classes
- `bilinear`: Use bilinear upsampling instead of transposed convolutions

---

## Modifications from Original

This fork includes the following changes:

1. **Apple Silicon Support**: Auto-detection and support for MPS device
2. **Device-agnostic code**: Works on CUDA, MPS, or CPU
3. **Memory handling**: Proper cache clearing for both CUDA and MPS
4. **Kaggle references removed**: No download scripts, custom data only

**Modified files:**
- `train.py` (lines 192-200, 81, 232-255)
- `predict.py` (lines 88-94)

---

## Known Limitations

- **AMP (Automatic Mixed Precision)** only works on CUDA, disabled on MPS
- **Pin memory** warning on MPS (PyTorch limitation, can be ignored)
- **Channels-last memory format** may have limited support on MPS in older PyTorch versions

---

## Credits

- **Original Implementation:** [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- **U-Net Paper:** Ronneberger et al., 2015
- **Apple Silicon Modifications:** 2025

---

## License

See [LICENSE](LICENSE) file (GNU GPLv3)
