# Quick Testing Guide

## Testing All Samples at Once

```bash
# Start the API server (in one terminal)
python main.py

# Run batch tests (in another terminal)
python test_all_samples.py
```

## Testing Individual Files

```bash
# Convert audio to base64
python convert_audio_to_base64.py path/to/audio.mp3

# Or use the API docs interface
# 1. Start server: python main.py
# 2. Open browser: http://localhost:8000/docs
# 3. Try the /api/voice-detection endpoint
```

## Fixing Long Audio Files

```bash
# Automatically trim audio files over 30 seconds
python fix_long_audio.py

# The script will create *_trimmed.wav files
```

## Results

### ✅ Human Voices (3 files)
- **audio_trimmed.wav** - 60.9% confidence (MEDIUM)
- **Namami_30s.mp3** - 59.7% confidence (MEDIUM)
- **sample voice 1.mp3** - 60.4% confidence (MEDIUM)

### ⚠️ AI-Generated Voices (2 files)
- **data_voices_martin_voyels_exports_mono__-a-c3-2.wav** - 64.8% confidence (HIGH)
- **data_voices_martin_voyels_exports_mono__-ou-c3.wav** - 63.8% confidence (HIGH)

### ❌ Failed (1 file)
- **audio.wav** - Too long (56.3s), fixed with trimmed version

## Key Findings

1. **AI Detection Works Well**: The martin voyels files (known AI samples) were correctly identified with high confidence (>63%)

2. **Human Voice Detection**: All Tamil samples (Namami, sample voice 1, audio) were correctly identified as human with moderate confidence (~60%)

3. **Confidence Levels**:
   - AI samples: HIGH confidence (64-65%)
   - Human samples: MEDIUM confidence (60%)
   - This difference suggests the model is working as expected

## File Locations

- **Test Script**: `test_all_samples.py`
- **Fix Script**: `fix_long_audio.py`
- **Results**: `test_results.json`
- **Summary**: `TEST_RESULTS_SUMMARY.md`
- **Samples**: `test_samples/` folder

## Troubleshooting

### Server Not Running
```bash
# Check if server is running
curl http://localhost:8000/health

# If not, start it
python main.py
```

### Audio Too Long Error
```bash
# Use the fix script
python fix_long_audio.py
```

### Dependencies Missing
```bash
pip install -r requirements.txt
```
