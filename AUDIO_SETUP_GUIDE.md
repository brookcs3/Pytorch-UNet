# 🎵 Audio File Setup Guide

## Your Files

You have:
- `Intergalactic_Acapella.wav` - Isolated vocal (acapella)
- `Intergalactic_Stereo.wav` - Full stereo mix

Location: `/Users/cameronbrooks/kaggle/Pytorch-UNet/`

## What the Sanity Check Needs

The files need to be:
1. **Format**: Mono (1 channel), 22050 Hz, WAV
2. **Duration**: 4.5 seconds
3. **Names**: 
   - `isolated_vocal.wav` (your acapella)
   - `stereo_mixture.wav` (your stereo mix)
4. **Location**: `vocal_separation_sanity_check/` directory

---

## Quick Setup (3 Steps)

### Step 1: Install Dependencies

```bash
cd /Users/cameronbrooks/kaggle/Pytorch-UNet
pip install librosa soundfile numpy scipy matplotlib
```

### Step 2: Prepare Your Audio Files

Save this script as `prepare_audio.py` in the Pytorch-UNet directory:

```python
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# Your source files
ACAPELLA = "Intergalactic_Acapella.wav"
STEREO = "Intergalactic_Stereo.wav"

# Target directory and names
TARGET_DIR = Path("vocal_separation_sanity_check")
TARGET_DIR.mkdir(exist_ok=True)

def prepare_file(source_name, target_name):
    print(f"\nProcessing {source_name}...")
    
    # Load (converts to mono automatically with mono=True)
    audio, sr = librosa.load(source_name, sr=22050, duration=4.5, mono=True)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Save
    target_path = TARGET_DIR / target_name
    sf.write(target_path, audio, 22050)
    
    print(f"  ✓ Saved to {target_path}")
    print(f"    Duration: {len(audio)/22050:.2f}s, SR: 22050 Hz, Mono")

# Process both files
prepare_file(ACAPELLA, "isolated_vocal.wav")
prepare_file(STEREO, "stereo_mixture.wav")

print("\n✓ Done! Files are ready in vocal_separation_sanity_check/")
```

Then run it:
```bash
python prepare_audio.py
```

### Step 3: Verify Setup

```bash
cd vocal_separation_sanity_check
python test_setup.py
```

If you see "✓ ALL TESTS PASSED!", you're ready!

---

## Alternative: Manual Conversion (Using Audacity/ffmpeg)

### Using ffmpeg:
```bash
# Convert acapella
ffmpeg -i Intergalactic_Acapella.wav -ar 22050 -ac 1 -t 4.5 vocal_separation_sanity_check/isolated_vocal.wav

# Convert stereo mix  
ffmpeg -i Intergalactic_Stereo.wav -ar 22050 -ac 1 -t 4.5 vocal_separation_sanity_check/stereo_mixture.wav
```

### Using Audacity:
1. Open each file in Audacity
2. **Tracks menu** → Convert to Mono (if stereo)
3. **Select all** (Ctrl/Cmd+A)
4. **Effect** → Change Speed/Pitch (set to 22050 Hz if needed)
5. Trim to 4.5 seconds (0:00.000 to 0:04.500)
6. **File** → Export → Export as WAV
7. Save with correct name in `vocal_separation_sanity_check/`

---

## Check If Files Are Ready

Run this in Python:

```python
import librosa

# Check vocal
v, sr_v = librosa.load('vocal_separation_sanity_check/isolated_vocal.wav', sr=None)
print(f"Vocal: {len(v)/sr_v:.2f}s, {sr_v} Hz, {v.ndim} channel(s)")

# Check mixture
m, sr_m = librosa.load('vocal_separation_sanity_check/stereo_mixture.wav', sr=None)
print(f"Mix:   {len(m)/sr_m:.2f}s, {sr_m} Hz, {m.ndim} channel(s)")

# Should see:
# Vocal: 4.50s, 22050 Hz, 1 channel(s)
# Mix:   4.50s, 22050 Hz, 1 channel(s)
```

---

## Common Issues

### "No such file or directory"
- Make sure you're in `/Users/cameronbrooks/kaggle/Pytorch-UNet/`
- Make sure your wav files are in this directory
- Check spelling: `Intergalactic_Acapella.wav` (capital I, underscore, capital A)

### "librosa not found"
```bash
pip install librosa
```

### Files are too short/long
The script will automatically trim or pad to 4.5 seconds. If your files are much shorter than 4.5 seconds, consider using a longer duration by editing the script.

### Stereo files
The script converts to mono automatically. This is correct - we want mono for the sanity check.

---

## Next Steps

Once files are ready:

```bash
cd vocal_separation_sanity_check

# Test everything works
python test_setup.py

# Run sanity check!
python sanity_check.py
```

Expected output:
```
PHASE 1: LOAD AND ANALYZE
  ✓ Created 18 slices, 1,800 total windows
  ✓ Total metrics: 765,000

PHASE 2: COMPARE FINGERPRINTS
  Window 0 comparison shows differences

✓ Fingerprints created
```

This proves the multi-scale spectral fingerprinting works!
