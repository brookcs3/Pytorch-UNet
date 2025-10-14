"""
Audio File Preparation Script
==============================
This script will:
1. Check your audio files
2. Convert them to the right format if needed
3. Copy them to the correct location with correct names
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

print("="*70)
print("AUDIO FILE PREPARATION")
print("="*70)

# File paths
ACAPELLA_SOURCE = Path("/Users/cameronbrooks/kaggle/Pytorch-UNet/Intergalactic_Acapella.wav")
STEREO_SOURCE = Path("/Users/cameronbrooks/kaggle/Pytorch-UNet/Intergalactic_Full.wav")

TARGET_DIR = Path("/Users/cameronbrooks/kaggle/Pytorch-UNet/vocal_separation_sanity_check")
TARGET_VOCAL = TARGET_DIR / "isolated_vocal.wav"
TARGET_MIXTURE = TARGET_DIR / "stereo_mixture.wav"

# Config
TARGET_SR = 22050  # Sample rate
TARGET_DURATION = 4.7  # seconds (ensures exactly 100 windows at hop_length=1024)

def check_and_prepare_file(source_path, target_path, file_type):
    """Check audio file and prepare it for sanity check"""
    
    print(f"\n[Processing {file_type}]")
    
    # Check if source exists
    if not source_path.exists():
        print(f"  ✗ Source file not found: {source_path}")
        print(f"    Please make sure the file exists at this location")
        return False
    
    print(f"  ✓ Found: {source_path.name}")
    
    # Load audio
    try:
        audio, sr = librosa.load(str(source_path), sr=None, mono=False)
        print(f"  ✓ Loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return False
    
    # Check format
    duration = len(audio.flatten()) / sr if audio.ndim == 1 else len(audio[0]) / sr
    channels = 1 if audio.ndim == 1 else audio.shape[0]
    
    print(f"  Current format:")
    print(f"    Sample rate: {sr} Hz")
    print(f"    Channels: {channels}")
    print(f"    Duration: {duration:.2f} seconds")
    
    # Prepare audio for sanity check
    print(f"\n  Preparing for sanity check...")
    
    # Convert to mono if stereo
    if audio.ndim == 2:
        audio = librosa.to_mono(audio)
        print(f"    ✓ Converted to mono")
    
    # Resample if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        print(f"    ✓ Resampled to {TARGET_SR} Hz")
    
    # Trim or pad to target duration
    target_samples = int(TARGET_DURATION * TARGET_SR)
    current_samples = len(audio)
    
    if current_samples > target_samples:
        # Trim to target duration
        audio = audio[:target_samples]
        print(f"    ✓ Trimmed to {TARGET_DURATION} seconds")
    elif current_samples < target_samples:
        # Pad with silence
        audio = np.pad(audio, (0, target_samples - current_samples))
        print(f"    ✓ Padded to {TARGET_DURATION} seconds")
    else:
        print(f"    ✓ Duration is correct ({TARGET_DURATION} seconds)")
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    print(f"    ✓ Normalized")
    
    # Save
    TARGET_DIR.mkdir(exist_ok=True)
    sf.write(str(target_path), audio, TARGET_SR)
    print(f"\n  ✓ Saved to: {target_path.name}")
    print(f"    Location: {target_path.parent}")
    
    return True

# Main execution
print("\nChecking source files...")
print(f"  Acapella:   {ACAPELLA_SOURCE}")
print(f"  Stereo mix: {STEREO_SOURCE}")

print("\nTarget directory:")
print(f"  {TARGET_DIR}")

# Process vocal
vocal_ok = check_and_prepare_file(ACAPELLA_SOURCE, TARGET_VOCAL, "Isolated Vocal")

# Process mixture
mixture_ok = check_and_prepare_file(STEREO_SOURCE, TARGET_MIXTURE, "Stereo Mixture")

# Summary
print("\n" + "="*70)
if vocal_ok and mixture_ok:
    print("✓ SUCCESS - Both files are ready!")
    print("="*70)
    print("\nYour files are prepared:")
    print(f"  Vocal:   {TARGET_VOCAL}")
    print(f"  Mixture: {TARGET_MIXTURE}")
    print("\nBoth files are:")
    print(f"  - Mono (1 channel)")
    print(f"  - {TARGET_SR} Hz sample rate")
    print(f"  - {TARGET_DURATION} seconds duration")
    print(f"  - Normalized")
    print("\nYou can now run:")
    print("  cd /Users/cameronbrooks/kaggle/Pytorch-UNet/vocal_separation_sanity_check")
    print("  python test_setup.py")
    print("  python sanity_check.py")
else:
    print("✗ FAILED - Please fix the issues above")
    print("="*70)
