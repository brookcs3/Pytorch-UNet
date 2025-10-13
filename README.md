# PyTorch U-Net (Historical Archive)

⚠️ **This is an OLD and DEFUNCT project** ⚠️

Originally created for the **Kaggle Carvana Image Masking Challenge** (2017). The competition dataset is no longer publicly available, but the code still works and has been updated for Apple Silicon.

**Status:** ✅ Working | ✅ Apple Silicon MPS support | ❌ Original dataset unavailable | ✅ Custom data supported

**Original Competition:** https://www.kaggle.com/c/carvana-image-masking-challenge (CLOSED - Achieved Dice coefficient 0.988423)

---

## What is U-Net?

A lightweight convolutional neural network for **semantic segmentation** (pixel-wise classification). Only **~126 lines of model code**, ~7.7M parameters.

**Use cases:** Medical imaging, background removal, satellite analysis, audio spectrograms, any pixel-wise task

**Architecture:** Encoder-decoder with skip connections | [Original Paper (2015)](https://arxiv.org/abs/1505.04597)

---

## Installation

```bash
cd /path/to/Pytorch-UNet
uv pip install -r requirements.txt
uv pip install torch torchvision
```

**Requirements:** Python 3.11+, PyTorch 2.0+

**Device priority:** MPS (Apple Silicon) → CUDA (NVIDIA) → CPU

---

## Quick Start

### Test Import & Model
```bash
python3 -c "from unet import UNet; m = UNet(3, 2); print(f'✅ {sum(p.numel() for p in m.parameters()):,} parameters')"
```

### Mini Training Example (Generates `.pth` file)
```python
import torch
from unet import UNet

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Fake data (replace with real data)
x = torch.rand(1, 3, 256, 256).to(device)
y = torch.randint(0, 2, (1, 256, 256)).to(device)

# Training loop
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "model.pth")
```

### Load Trained Model
```python
model = UNet(n_channels=3, n_classes=2).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

---

## Training with Custom Data

**Data structure:**
```
data/
├── imgs/          # RGB images (any size)
└── masks/         # Grayscale masks (same filename + _mask suffix)
```

**Training:**
```bash
python3 train.py --epochs 10 --batch-size 2 --scale 0.5 --validation 20
```

**Prediction:**
```bash
python3 predict.py -i input.jpg -o output_mask.png
```

---

## Architecture Overview

**Files:** `unet/unet_model.py` (48 lines) + `unet/unet_parts.py` (78 lines) = **126 lines total**

### Components

**Building blocks** (`unet_parts.py`):
- `DoubleConv`: Conv2d(3×3) → BatchNorm → ReLU (×2)
- `Down`: MaxPool2d → DoubleConv (encoder)
- `Up`: Upsample → Concat with skip → DoubleConv (decoder)
- `OutConv`: 1×1 Conv for final output

**Network structure** (`unet_model.py`):
```python
# Encoder (downsampling)
inc:   3 → 64   [H×W]       ────┐
down1: 64 → 128 [H/2×W/2]   ────┼──┐
down2: 128 → 256 [H/4×W/4]  ────┼──┼──┐
down3: 256 → 512 [H/8×W/8]  ────┼──┼──┼──┐
down4: 512 → 1024 [H/16×W/16]   │  │  │  │  (bottleneck)

# Decoder (upsampling with skip connections)
up1: 1024 → 512 [H/8×W/8]  ←────┘  │  │  │
up2: 512 → 256 [H/4×W/4]   ←───────┘  │  │
up3: 256 → 128 [H/2×W/2]   ←──────────┘  │
up4: 128 → 64 [H×W]        ←─────────────┘
outc: 64 → n_classes [H×W]
```

**Key insight:** Skip connections preserve spatial details from encoder for precise segmentation.

**Locations:**
- Encoder: `unet_model.py:13-18`
- Decoder: `unet_model.py:19-22`
- Skip connections: `unet_model.py:31-34`
- Building blocks: `unet_parts.py:8-77`

---

## Understanding `.pth` Files

### What Training Creates

| Component | Location | Created By |
|-----------|----------|------------|
| **Architecture** (structure) | Python code | Written by developers |
| **Weights** (learned knowledge) | `.pth` file (~30 MB) | Training process |
| **Functional model** | Both combined | Code + weights |

**Analogy:** Code = brain blueprint, `.pth` = learned memories. Need both to function.

### What's in a `.pth` File?

~7.7M learned parameter values (weights + biases) as a dictionary:
```python
state_dict = torch.load("model.pth")
# Contains: "inc.double_conv.0.weight", "down1.maxpool_conv.1.weight", etc.
```

**Important:** `.pth` contains ONLY numbers, not architecture. You need the original code to load it.

### Common Operations

```python
# Save weights only (recommended)
torch.save(model.state_dict(), "weights.pth")

# Save full checkpoint (training resume)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, "checkpoint.pth")

# Load checkpoint
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## Adapting for Audio Separation

This vanilla U-Net works well for audio (Spleeter uses similar architecture). Minor tweaks for optimization:

### Key Changes
```python
# 1. Input/output channels
model = UNet(n_channels=1, n_classes=4)  # 1 spectrogram → 4 stems

# 2. Output activation (for masks)
return torch.sigmoid(self.conv(x))  # Bounds to [0,1]

# 3. Loss function
criterion = nn.L1Loss()  # Instead of CrossEntropyLoss
```

### Optional Optimizations

| Component | Current | Audio Alternative | Why |
|-----------|---------|-------------------|-----|
| Activation | `ReLU` | `LeakyReLU(0.2)` | Handles negatives better |
| Normalization | `BatchNorm2d` | `InstanceNorm2d` or `GroupNorm` | Better for variable lengths |
| Downsampling | `MaxPool2d` | Strided `Conv2d` | Learnable |
| Output | Raw logits | `sigmoid` or `relu` | Bounded masks |

**Note:** Spleeter achieves excellent results with vanilla U-Net. Optimizations are refinements, not requirements.

---

## Why This Works

**Minimal codebase isn't a limitation—it's a feature:**
- 126 lines = complete production-ready architecture
- Same fundamentals as Logic Pro, Serato, iZotope stem splitters
- Difference from commercial products: training data quality/quantity, not architecture
- Works for images, audio, medical scans, satellite imagery

**Reality check:** Logic Pro's excellent stem separation comes from massive high-quality training data, not architectural magic. The U-Net is likely very similar to this implementation.

---

## Modifications from Original

**Changes in this fork:**
1. Apple Silicon MPS support (auto-detection)
2. Device-agnostic (CUDA/MPS/CPU)
3. Proper memory cache clearing
4. Kaggle references removed

**Modified files:** `train.py:81,192-200,232-255`, `predict.py:88-94`

---

## Known Limitations

- AMP (mixed precision) only works on CUDA, not MPS
- Pin memory warnings on MPS (PyTorch limitation, safe to ignore)
- Channels-last format limited on MPS in older PyTorch versions

---

## Credits & License

**Original:** [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) | **Paper:** Ronneberger et al., 2015 | **Apple Silicon mods:** 2025

**License:** GNU GPLv3 (see [LICENSE](LICENSE))
