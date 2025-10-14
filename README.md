# PyTorch U-Net + Vocal Separation Experiments

Old U-Net implementation from the 2017 Kaggle Carvana Image Masking Challenge. Still works, been updated for Apple Silicon. Also added some vocal separation experiments to understand what U-Net actually learns before training it.

**Original competition:** https://www.kaggle.com/c/carvana-image-masking-challenge (closed, achieved 0.988 Dice coefficient)

---

## Documentation

- **Main README** - This file (overview)
- **[Vocal Separation Guide](vocal_separation_sanity_check/README.md)** - How to use the vocal separation experiments
- **[Technical Details](vocal_separation_sanity_check/COMPLETE_DOC.md)** - Deep dive into how it works
- **[Audio Setup Guide](AUDIO_SETUP_GUIDE.md)** - Help with audio file preparation
- **[Push to GitHub](PUSH_TO_GITHUB.md)** - Instructions for sharing this repo

---

## What's in Here

### U-Net Implementation
Standard encoder-decoder with skip connections. About 126 lines of model code, 7.7M parameters. Based on the [2015 Ronneberger paper](https://arxiv.org/abs/1505.04597).

Works for image segmentation, medical imaging, audio spectrograms, whatever you want to segment.

### Vocal Separation Proof-of-Concept (New)
Experiments in `vocal_separation_sanity_check/` directory. Trying to manually separate vocals from a mix without training a neural network, just to prove the approach works before spending days training.

Gets about 70-80% quality with manual spectral fingerprinting. If that works, then training a U-Net should get to 95%+ automatically.

**Quick start:**
```bash
cd vocal_separation_sanity_check

# Add your audio files to process/100-window/:
#   yourfile_100-full.wav (full mix)
#   yourfile_100-stem.wav (vocal only)

python prepare_audio_files.py
python sanity_check_complete.py
```

See [vocal_separation_sanity_check/README.md](vocal_separation_sanity_check/README.md) for details.

---

## Quick Start

### U-Net

Install stuff:
```bash
pip install torch torchvision numpy librosa soundfile scipy matplotlib
```

Test it works:
```python
from unet import UNet
model = UNet(n_channels=3, n_classes=2)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Vocal Separation

```bash
cd vocal_separation_sanity_check

# Add your files to the right directory:
# - For quick test: process/100-window/yourfile_100-full.wav and yourfile_100-stem.wav
# - For full song: process/no-limit/yourfile_nl-full.wav and yourfile_nl-stem.wav

python prepare_audio_files.py

# Then run:
python sanity_check_complete.py        # for 100-window version
# OR
python sanity_check_full_length.py     # for no-limit version
```

Outputs separated vocal to `output/extracted_vocal.wav` or `output_full/extracted_vocal_full.wav`.

---

## U-Net Architecture

Simple encoder-decoder:

```
Encoder (down):
  input → 64 → 128 → 256 → 512 → 1024 (bottleneck)

Decoder (up, with skip connections):
  1024 → 512 → 256 → 128 → 64 → output
```

Skip connections concatenate features from encoder to decoder at each level. Preserves spatial details.

Code in `unet/unet_model.py` and `unet/unet_parts.py`.

---

## Training on Custom Data

Image segmentation:

```bash
# Data structure:
# data/imgs/ - input images
# data/masks/ - segmentation masks

python train.py --epochs 10 --batch-size 2
```

For audio separation:
- Use spectrograms as input (1 channel)
- Use source masks as output (4 channels for drums/bass/vocals/other)
- Change loss to L1Loss or MSE instead of CrossEntropyLoss
- Add sigmoid output activation

See vocal separation experiments for a working example of the spectral approach.

---

## Device Support

Auto-detects best available:
- NVIDIA GPU (CUDA) on Windows/Linux
- Apple Silicon (MPS) on Mac
- CPU fallback

```python
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```

---

## Files

### Core U-Net
- `unet/unet_model.py` - main model (48 lines)
- `unet/unet_parts.py` - building blocks (78 lines)
- `train.py` - training script
- `predict.py` - inference script
- `evaluate.py` - evaluation metrics

### Vocal Separation
- `vocal_separation_sanity_check/` - proof-of-concept experiments
  - `process/100-window/` - place your files here for quick test
  - `process/no-limit/` - place your files here for full song
  - `prepare_audio_files.py` - processes and moves your files
  - `sanity_check_complete.py` - runs 100-window version
  - `sanity_check_full_length.py` - runs full-song version
  - `test_setup.py` - verify installation
  - `README.md` - detailed usage guide
  - `COMPLETE_DOC.md` - technical deep dive
- `AUDIO_SETUP_GUIDE.md` - guide for setting up audio files
- `PUSH_TO_GITHUB.md` - sharing instructions

---

## Vocal Separation Approach

Manual implementation of what U-Net should learn:

1. Create 18 different "views" of the spectrogram (conv filters)
2. Compress each through encoder layers to bottleneck
3. Extract 425 metrics per time window (400-point frequency profile + 25 features)
4. Optimize mixture parameters to match vocal fingerprint
5. Reconstruct separated audio using learned parameters

Takes a few minutes, gets 70-80% quality. Proves the concept works.

Then train U-Net to do it automatically in 10ms at 95%+ quality.

---

- 

---

## Credits

**Original U-Net:** [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)  
**Paper:** Ronneberger et al., 2015  
**Apple Silicon updates:** 2025  
**Vocal separation experiments:** 2025  

**License:** GNU GPLv3
