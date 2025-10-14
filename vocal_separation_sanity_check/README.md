# Vocal Separation Sanity Check

Trying to prove that you can separate vocals from a mix without actually training a neural network. Just wanted to see if the core idea works before spending days training a U-Net.

## The Idea

Basically: if I can manually extract a vocal by creating a unique enough "fingerprint" of what a vocal looks like across 18 different views of the spectrogram, then a U-Net should be able to learn to do the same thing automatically (and way faster).

This is the manual proof-of-concept version. Takes a few minutes to process a song, gets maybe 70-80% quality. But if this works at all, then training a U-Net on thousands of examples should get us to 95%+ and do it in 10ms instead of minutes.

## How It Works

1. Take the isolated vocal and the full mix
2. Create 18 different "slices" of each spectrogram (different conv filters basically - horizontal lines, vertical lines, diagonals, edges, etc)
3. Run each slice through 4 encoder layers to compress it down to a bottleneck
4. At the bottleneck, extract a ton of metrics - 400-point frequency profile plus 25 other features
5. Now you have ~765,000 unique measurements describing the vocal
6. Try to adjust the mixture's spectrogram until its fingerprint matches the vocal's
7. Apply those adjustments, convert back to audio

If the fingerprint is unique enough, the only way the mixture can match it is to actually sound like the vocal.

## The 18 Slices

Different conv2d filters to catch different patterns:

- slice_0: raw magnitude (baseline)
- slice_1-8: basic patterns (horizontal/vertical/diagonal/blob/harmonic/edge detection)
- slice_9-15: oriented edge detectors at different angles (22.5°, 45°, 67.5°, etc)
- slice_16-18: laplacian and pooled versions

Vocals have specific patterns - sustained frequencies, harmonic stacks, mid-range energy, formants. These 18 views capture that.

## Setup

Need these:
```bash
uv pip install numpy librosa soundfile scipy matplotlib
```

You'll need two audio files:
- `isolated_vocal.wav` - acapella or isolated vocal track
- `stereo_mixture.wav` - the full song with everything

Just drop them in this directory.

## Running It

Three versions depending on what you want:

### Quick test (4.7 seconds, ~100 windows)
```bash
python sanity_check_complete.py
```
Takes about 3 minutes. Good for testing if it works at all.

### Full song
```bash
python sanity_check_full_length.py
```
Processes the entire audio file. Takes longer obviously - maybe 30-50 minutes for a 3 minute song. Outputs to `output_full/` so it doesn't conflict with the test version.

### Just analysis (no audio output)
```bash
python sanity_check.py
```
This one only does phases 1-2 (fingerprint creation and comparison). Doesn't actually generate audio. Useful for debugging.

## What You Get

After running complete version:

```
output/
├── extracted_vocal.wav              # the separated vocal
├── 1_original_mixture.wav           # copy of input for comparison
├── 2_target_vocal.wav               # copy of target for comparison
├── optimization_loss.png            # shows the learning curve
└── spectrograms.png                 # visual comparison
```

Full-length version puts everything in `output_full/` instead.

## Expected Results

Not expecting perfection. This is a proof of concept.

**Should work:**
- Vocal is audible and mostly clear
- Drums are way quieter
- Bass is way quieter
- You can tell it's the vocal

**Won't work:**
- Studio-quality isolation
- Zero artifacts (there will be some clicking/phasing)
- Complete removal of all other instruments
- Perfect timbre preservation

If we hit 70-80% quality, that proves the concept. Then U-Net training should get us to 95%+.

## The Fingerprint

Each time window (every ~46ms) gets:
- 400-point frequency profile (energy at 0Hz, 50Hz, 100Hz, ... 20kHz)
- 6 band energies (bass, low-mid, mid, high-mid, presence, high)
- 6 spectral shape metrics (centroid, spread, rolloff, flatness, slope, crest)
- 5 harmonic features (fundamental freq, num harmonics, spacing, strength, etc)
- 4 formant measurements
- 4 dynamics measurements

= 425 metrics per window

With 18 slices and ~100 windows per slice, that's around 765,000 data points describing the vocal. Should be unique enough that only the actual vocal matches.

## Files

- `sanity_check.py` - phases 1-2 only (analysis, no output audio)
- `sanity_check_complete.py` - full pipeline, 4.7 second snippet
- `sanity_check_full_length.py` - full pipeline, entire song
- `prepare_audio_files.py` - helper to convert your audio to the right format
- `test_setup.py` - verify everything is installed correctly
- `requirements.txt` - dependencies

## Troubleshooting

**"FileNotFoundError"**
Run `prepare_audio_files.py` first. It'll convert your audio files and put them in the right place.

**Takes forever**
Use the complete version instead of full-length. Or reduce the number of iterations in the config.

**Out of memory**
Reduce n_fft from 2048 to 1024 in the config.

**Results sound terrible**
Make sure your input files are correct. The "Full" file should be the complete mix, and the "Acapella" file should be just the vocal. If they're swapped or wrong, it won't work.

## What This Proves

If this works even a little bit, it means:
1. Multi-scale spectral fingerprinting is a valid approach
2. The encoder-decoder architecture makes sense
3. U-Net should be able to learn this automatically
4. Training a U-Net on this task is worth the effort

If it doesn't work, then either the fingerprint isn't unique enough, or the optimization approach is wrong, or the whole concept is flawed. Better to find out in a few minutes of manual work than after days of training.

## Next Steps

Assuming this works:
1. Figure out which slices were most useful
2. Design a U-Net that mirrors this architecture
3. Gather a dataset of thousands of (vocal, mixture) pairs
4. Train the U-Net
5. Compare trained model to this baseline

The trained model should be way better (95%+ quality) and way faster (10ms vs minutes).

## Notes

This is not production-ready. It's a proof of concept. For actual vocal separation use Spleeter or Demucs or something.

The point is to understand what's actually happening before throwing a neural network at it. Makes the architecture decisions way more informed.
