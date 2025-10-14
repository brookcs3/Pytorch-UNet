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

# Go to vocal separation directory
cd vocal_separation_sanity_check

# Add audio files to the right directory:
# Option 1: Quick test (100 windows)
#   Copy files to: process/100-window/
#   Name them: yourfile_100-full.wav and yourfile_100-stem.wav

# Option 2: Full song
#   Copy files to: process/no-limit/
#   Name them: yourfile_nl-full.wav and yourfile_nl-stem.wav

# Prepare the files
python prepare_audio_files.py

# Run the sanity check
python sanity_check_complete.py       # for 100-window
# OR
python sanity_check_full_length.py    # for no-limit

# Check output
ls output/          # for 100-window version
# OR
ls output_full/     # for no-limit version
```

## Quick Test

They can verify everything works:

```bash
cd vocal_separation_sanity_check
python test_setup.py
```

Should see: `✓ ALL TESTS PASSED!`

## File Structure After Cloning

```
pytorch-unet-vocal-separation/
├── README.md                           # Main overview
├── PUSH_TO_GITHUB.md                   # This file
├── AUDIO_SETUP_GUIDE.md                # Audio setup help
│
├── unet/                               # Core U-Net implementation
│   ├── unet_model.py
│   └── unet_parts.py
│
├── train.py
├── predict.py
├── evaluate.py
│
└── vocal_separation_sanity_check/
    ├── README.md                       # Usage guide
    ├── COMPLETE_DOC.md                 # Technical details
    │
    ├── process/                        # Users add files here
    │   ├── 100-window/
    │   │   └── README.md               # Instructions for this dir
    │   └── no-limit/
    │       └── README.md               # Instructions for this dir
    │
    ├── prepare_audio_files.py          # Processes user files
    ├── sanity_check_complete.py        # 100-window version
    ├── sanity_check_full_length.py     # No-limit version
    └── test_setup.py                   # Verify installation
```

## Workflow Summary

1. Clone repo
2. Install dependencies
3. Add audio files to `process/100-window/` or `process/no-limit/`
4. Name files correctly (`*_100-full.wav` + `*_100-stem.wav` OR `*_nl-full.wav` + `*_nl-stem.wav`)
5. Run `prepare_audio_files.py`
6. Run appropriate sanity check script
7. Check `output/` or `output_full/` for results

## Common Issues

**"Permission denied (publickey)"**
→ Use HTTPS URL instead of SSH, or set up SSH keys

**"Already exists" error**
→ The repo already has commits, use `git pull origin main` first

**"NO FILES FOUND" when running prepare script**
→ Files are in wrong directory or not named correctly. Check the README in the process directory.

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
- README files in process directories ARE tracked (to help users)
