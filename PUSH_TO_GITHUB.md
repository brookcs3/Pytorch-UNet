# Pushing to GitHub

Quick guide to get this on GitHub so the team can use it.

## Initial Setup (if not already a git repo)

```bash
cd /Users/cameronbrooks/kaggle/Pytorch-UNet

# Check if it's already a repo
git status

# If not, initialize it
git init
git add .
git commit -m "Initial commit with vocal separation experiments"
```

## Create GitHub Repo

1. Go to https://github.com/new
2. Create a new repository (public or private)
3. Don't initialize with README (we already have one)
4. Copy the repository URL

## Push to GitHub

```bash
# Add the remote (replace with your actual repo URL)
git remote add origin https://github.com/yourusername/pytorch-unet-vocal-separation.git

# Or if using SSH:
git remote add origin git@github.com:yourusername/pytorch-unet-vocal-separation.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## For Your Team to Use

They can clone and run:

```bash
# Clone the repo
git clone https://github.com/yourusername/pytorch-unet-vocal-separation.git
cd pytorch-unet-vocal-separation

# Install dependencies
pip install torch torchvision numpy librosa soundfile scipy matplotlib

# Try the vocal separation
cd vocal_separation_sanity_check
pip install -r requirements.txt

# Add their audio files
# - isolated_vocal.wav (acapella)
# - stereo_mixture.wav (full mix)

# Or use the preparation script
cd ..
# Edit prepare_audio_files.py to point to their files
python prepare_audio_files.py

# Run the sanity check
cd vocal_separation_sanity_check
python sanity_check_complete.py

# Check output
ls output/
# Should see: extracted_vocal.wav
```

## Quick Test

They can verify everything works:

```bash
cd vocal_separation_sanity_check
python test_setup.py
```

Should see: `✓ ALL TESTS PASSED!`

## What's Included

After cloning, they get:

### Core U-Net
- `unet/` - model code
- `train.py` - training script
- `predict.py` - inference
- `evaluate.py` - evaluation

### Vocal Separation Experiments
- `vocal_separation_sanity_check/` - proof-of-concept code
- `sanity_check.py` - phases 1-2 only
- `sanity_check_complete.py` - full pipeline, short clip
- `sanity_check_full_length.py` - full pipeline, entire song
- `prepare_audio_files.py` - audio prep helper
- `test_setup.py` - dependency checker
- `README.md` - usage guide
- `COMPLETE_DOC.md` - technical details

### Docs
- `README.md` - main readme
- `AUDIO_SETUP_GUIDE.md` - audio file setup
- `vocal_separation_sanity_check/README.md` - sanity check guide

## Common Issues

**"Permission denied (publickey)"**
→ Use HTTPS URL instead of SSH, or set up SSH keys

**"Already exists" error**
→ The repo already has commits, use `git pull origin main` first

**Audio files in repo**
→ They're gitignored, which is correct. Each user adds their own audio files.

## Updating Later

When you make changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

Team members pull updates:

```bash
git pull
```

## Notes

- Audio files (`.wav`) are gitignored - users supply their own
- Output directories are gitignored - generated locally
- Model checkpoints (`.pth`) are gitignored - too large for git
