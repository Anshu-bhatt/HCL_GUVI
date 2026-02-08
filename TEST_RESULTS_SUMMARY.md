# Audio Sample Test Results
**Test Date:** February 3, 2026  
**API:** AI Voice Detection API (Wav2Vec2 + Acoustic Features)

## Summary

Tested **6 audio files** (including trimmed version) from the `test_samples` folder using the AI Voice Detection API.

### Results Overview

| Sample | Classification | Confidence | Level | Status |
|--------|---------------|------------|-------|--------|
| audio.wav | ❌ ERROR | - | - | Audio too long (56.3s, max 30s) |
| audio_trimmed.wav | ✅ **HUMAN** | 60.9% | MEDIUM | ✅ Success |
| data_voices_martin_voyels_exports_mono__-a-c3-2.wav | ⚠️ **AI-GENERATED** | 64.8% | HIGH | ✅ Success |
| data_voices_martin_voyels_exports_mono__-ou-c3.wav | ⚠️ **AI-GENERATED** | 63.8% | HIGH | ✅ Success |
| Namami_30s.mp3 | ✅ **HUMAN** | 59.7% | MEDIUM | ✅ Success |
| sample voice 1.mp3 | ✅ **HUMAN** | 60.4% | MEDIUM | ✅ Success |

### Statistics
- **Total Tested:** 6
- **Successfully Processed:** 5 (83%)
- **Human Voices:** 3 (60%)
- **AI-Generated Voices:** 2 (40%)
- **Uncertain:** 0 (0%)
- **Failed:** 1 (17% - due to length limit)

## Detailed Results

### 1. audio.wav (Original)
- **Status:** ❌ ERROR
- **Reason:** Audio too long (56.3 seconds, maximum allowed is 30 seconds)
- **File Size:** 5,406,764 bytes
- **Action Taken:** Created trimmed version (audio_trimmed.wav)

### 2. audio_trimmed.wav (Fixed)
- **Classification:** ✅ HUMAN
- **Confidence:** 60.9% (MEDIUM)
- **Duration:** 30.00 seconds
- **File Size:** 2,880,044 bytes
- **Explanation:** Detected as human voice with moderate confidence (61%) due to irregular acoustic patterns
- **Verdict:** This appears to be an authentic human voice recording

### 3. data_voices_martin_voyels_exports_mono__-a-c3-2.wav
- **Classification:** ⚠️ AI-GENERATED
- **Confidence:** 64.8% (HIGH)
- **Duration:** 1.23 seconds
- **Details:**
  - Wav2Vec2 Score: 0.8000
  - Acoustic Score: 0.2933
  - Combined Score: 0.6480
  - Processing Time: 584ms
- **Explanation:** Detected as AI-generated with very high confidence (65%) due to high consistency in voice embeddings
- **Verdict:** This appears to be synthetically generated speech

### 4. data_voices_martin_voyels_exports_mono__-ou-c3.wav
- **Classification:** ⚠️ AI-GENERATED
- **Confidence:** 63.8% (HIGH)
- **Duration:** 1.18 seconds
- **Details:**
  - Wav2Vec2 Score: 0.8000
  - Acoustic Score: 0.2604
  - Combined Score: 0.6381
  - Processing Time: 568ms
- **Explanation:** Detected as AI-generated with very high confidence (64%) due to high consistency in voice embeddings
- **Verdict:** This appears to be synthetically generated speech

### 5. Namami_30s.mp3
- **Classification:** ✅ HUMAN
- **Confidence:** 59.7% (MEDIUM)
- **Duration:** 30.00 seconds
- **Details:**
  - Wav2Vec2 Score: 0.5000
  - Acoustic Score: 0.1765
  - Combined Score: 0.4030
  - Processing Time: 718ms
- **Explanation:** Detected as human voice with moderate confidence (60%) due to irregular acoustic patterns
- **Verdict:** This appears to be an authentic human voice recording

### 6. sample voice 1.mp3
- **Classification:** ✅ HUMAN
- **Confidence:** 60.4% (MEDIUM)
- **Duration:** 23.59 seconds
- **Details:**
  - Wav2Vec2 Score: 0.5000
  - Acoustic Score: 0.1529
  - Combined Score: 0.3959
  - Processing Time: 833ms
- **Explanation:** Detected as human voice with moderate confidence (60%) due to irregular acoustic patterns
- **Verdict:** This appears to be an authentic human voice recording

## Changes Made

### 1. Created `test_all_samples.py`
A comprehensive batch testing script that:
- ✅ Automatically detects all audio files in test_samples folder
- ✅ Converts audio to Base64 format
- ✅ Sends detection requests to the API
- ✅ Displays detailed results for each sample
- ✅ Generates a summary report
- ✅ Saves results to JSON file

### 2. Created `fix_long_audio.py`
A utility script to handle audio files that exceed the 30-second limit:
- ✅ Automatically trims audio files to 30 seconds
- ✅ Preserves audio quality
- ✅ Successfully trimmed audio.wav (56.3s → 30.0s)
- ✅ Created audio_trimmed.wav which was successfully tested

### 3. Fixed Dependencies
- ✅ Installed missing Python packages (librosa, soundfile, torch, torchaudio, pydub, etc.)
- ✅ Upgraded torchvision to v0.25.0 to match torch v2.10.0
- ✅ Resolved all compatibility issues

### 4. API Server
- ✅ Successfully started the AI Voice Detection API on port 8000
- ✅ Models loaded (Wav2Vec2 + Acoustic Features)
- ✅ Server is running and processing requests

## Recommendations

### Audio File Guidelines
- ✅ Keep audio files under 30 seconds
- ✅ Use WAV or MP3 format
- ✅ Ensure good audio quality for accurate detection
- ✅ For longer files, use the `fix_long_audio.py` script to trim them

### Test Coverage
Consider adding more test samples with:
- Different languages (Hindi, Malayalam, Telugu)
- Various audio qualities
- Different speaking styles
- Background noise variations
- Multiple speakers
- Different AI voice generators for comparison

## Files Created

1. **test_all_samples.py** - Batch testing script for all audio samples
2. **fix_long_audio.py** - Utility to trim audio files exceeding 30 seconds
3. **test_trimmed_audio.py** - Quick test script for the trimmed audio
4. **TEST_RESULTS_SUMMARY.md** - This comprehensive report
5. **test_results.json** - Detailed JSON results from batch testing

## Next Steps

1. ✅ All audio samples have been tested
2. ✅ Long audio file fixed and tested (audio_trimmed.wav)
3. ✅ Results documented and saved
4. 🔄 Consider adding more diverse test samples
5. 🔄 Run periodic tests to validate model accuracy
6. 🔄 Monitor false positive/negative rates with known samples

## Test Script Location
**Path:** `D:\development\workspace\HCL\HCL_GUVI\test_all_samples.py`

**Usage:**
```bash
# Make sure server is running first
python main.py

# Then in another terminal:
python test_all_samples.py
```

## API Endpoint
- **URL:** http://localhost:8000
- **Detection Endpoint:** POST /api/voice-detection
- **Health Check:** GET /health
- **Docs:** http://localhost:8000/docs
