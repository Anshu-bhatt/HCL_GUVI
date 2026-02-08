# AI Voice Detection API - VoiceGUARD

## 📋 Project Overview

Multi-language AI voice detection system that identifies whether audio is **AI-generated** (deepfake) or **human-spoken** across 5 languages: Tamil, English, Hindi, Malayalam, and Telugu.

**Hackathon:** GUVI AI Hackathon  
**Problem Statement:** AI-Generated Voice Detection (Multi-Language)  
**Timeline:** 7 days  
**Current Status:** Feature 3 Complete ✓ (VoiceGUARD Model Integrated)

---

## 🎯 Project Goals

- ✅ Build REST API for voice authentication detection
- ✅ Support 5 languages (Tamil, English, Hindi, Malayalam, Telugu)
- ✅ Accept Base64-encoded audio (MP3, WAV, FLAC, OGG, M4A)
- ✅ Return classification (AI_GENERATED or HUMAN) with confidence score
- ✅ Pre-trained VoiceGUARD model integration
- ⏳ Deploy production-ready API

---

## 🏗️ Architecture & Strategy

### **Chosen Approach:** VoiceGUARD Pre-trained Model

**Why this approach:**

- 🎯 **Pre-trained model:** Uses `Mrkomiljon/voiceGUARD` from HuggingFace
- 🔥 **No training needed:** Fine-tuned for deepfake detection
- 📊 **High accuracy:** Trained on real deepfake datasets
- 🌍 **Language agnostic:** Works across all supported languages

### **Technology Stack**

```
┌─────────────────────────────────────────────────┐
│             FastAPI REST API                    │
│         POST /classify endpoint                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌────────────────────────────────────────┐     │
│  │     VoiceGUARD (Wav2Vec2-based)        │     │
│  │     Mrkomiljon/voiceGUARD              │     │
│  │     Fine-tuned for Deepfake Detection  │     │
│  └────────────────────────────────────────┘     │
│                    ↓                            │
│  ┌────────────────────────────────────────┐     │
│  │        Classification Output           │     │
│  │   • AI_GENERATED or HUMAN              │     │
│  │   • Confidence Score (0-100%)          │     │
│  │   • Confidence Level (HIGH/MEDIUM/LOW) │     │
│  │   • Detailed Probabilities             │     │
│  └────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

### **Core Components**

| Component               | File                     | Description                                   |
| ----------------------- | ------------------------ | --------------------------------------------- |
| **VoiceGUARD Detector** | `voiceguard_detector.py` | Pre-trained Wav2Vec2 model for classification |
| **Audio Processor**     | `audio_preprocessor.py`  | Base64 decoding, validation, preprocessing    |
| **API Models**          | `api_models.py`          | Pydantic request/response schemas             |
| **FastAPI Server**      | `main.py`                | REST API with `/classify` endpoint            |
| **Demo Script**         | `demo_classify.py`       | Test classification locally                   |

---

## ✅ Features Implemented

### Feature 1: Project Setup ✓

- Virtual environment setup
- Dependency installation
- Configuration management
- Basic FastAPI application

### Feature 2: Audio Processing Pipeline ✓

- Base64 audio decoding
- Multi-format support (MP3, WAV, FLAC, OGG, M4A)
- Audio preprocessing (normalization, trimming)
- Feature extraction (40+ acoustic features)

### Feature 3: VoiceGUARD Model Integration ✓ (NEW)

- **Pre-trained model:** `Mrkomiljon/voiceGUARD` from HuggingFace
- **Classification endpoint:** `POST /classify`
- **Confidence scoring:** 0-100% with levels (VERY_HIGH, HIGH, MEDIUM, LOW, VERY_LOW)
- **Detailed response:** Probabilities, raw scores, model info
- **Comprehensive tests:** Unit tests, API tests, demo script

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 2GB free disk space (for model cache)
- Stable internet connection (for first model download)

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd ai-voice-detection

# 2. Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py
```

### Running the API

```bash
# Start the FastAPI server
python main.py

# Server runs at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

### Demo Classification

```bash
# Run demo with synthetic audio
python demo_classify.py

# Classify a specific audio file
python demo_classify.py test_samples/sample_voice.mp3
```

---

## 📡 API Reference

### Endpoints

| Method | Endpoint      | Description        |
| ------ | ------------- | ------------------ |
| GET    | `/`           | API information    |
| GET    | `/health`     | Health check       |
| GET    | `/model/info` | Model information  |
| POST   | `/classify`   | **Classify audio** |

### POST /classify

Classify audio as AI-generated or Human.

#### Request

```json
{
  "audio_base64": "UklGRiQA...(base64 encoded audio)...",
  "language": "English" // Optional
}
```

#### Response

```json
{
  "classification": "AI_GENERATED",
  "confidence": 0.9234,
  "confidence_level": "VERY_HIGH",
  "probabilities": {
    "HUMAN": 0.0766,
    "AI_GENERATED": 0.9234
  },
  "description": "Synthetic/AI-generated voice - potential deepfake",
  "details": {
    "model": "Mrkomiljon/voiceGUARD",
    "audio_duration_seconds": 3.5,
    "sample_rate": 16000,
    "device": "cpu",
    "threshold_used": 0.5,
    "raw_scores": {
      "bonafide_score": 0.0766,
      "spoof_score": 0.9234
    }
  }
}
```

### Python Example

```python
import requests
import base64

# Read and encode audio file
with open("audio.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/classify",
    json={"audio_base64": audio_base64}
)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL Example

```bash
# Encode audio file
AUDIO_BASE64=$(base64 -i audio.mp3)

# Make request
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$AUDIO_BASE64\"}"
```

---

## 📊 Confidence Levels

| Level     | Confidence Range | Interpretation                       |
| --------- | ---------------- | ------------------------------------ |
| VERY_HIGH | 95-100%          | Very confident classification        |
| HIGH      | 85-95%           | High confidence                      |
| MEDIUM    | 70-85%           | Moderate confidence                  |
| LOW       | 55-70%           | Low confidence, may need review      |
| VERY_LOW  | <55%             | Uncertain, manual review recommended |

---

## 🧪 Testing

### Run All Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_voiceguard.py -v
pytest tests/test_api.py -v
```

### Test Coverage

| Test File                       | Coverage                       |
| ------------------------------- | ------------------------------ |
| `tests/test_voiceguard.py`      | VoiceGUARD detector unit tests |
| `tests/test_api.py`             | FastAPI endpoint tests         |
| `tests/test_audio_processor.py` | Audio processing tests         |
| `tests/test_model_detector.py`  | Legacy detector tests          |

---

## 📁 Project Structure

```
ai-voice-detection/
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 config.py                    # Configuration ✓
│
├── 🚀 API & Detection
├── 📄 main.py                      # FastAPI application ✓
├── 📄 voiceguard_detector.py       # VoiceGUARD model ✓ (NEW)
├── 📄 api_models.py                # Pydantic schemas ✓ (NEW)
├── 📄 demo_classify.py             # Demo script ✓ (NEW)
│
├── 🔊 Audio Processing
├── 📄 audio_preprocessor.py        # Audio processing ✓
├── 📄 process_audio.py             # CLI audio processor ✓
│
├── 🤖 Models (Legacy)
├── 📄 model_detector.py            # Wav2Vec2 heuristic detector
│
├── 📁 utils/
│   └── 📄 audio_helpers.py         # Helper functions ✓
│
├── 📁 tests/
│   ├── 📄 test_voiceguard.py       # VoiceGUARD tests ✓ (NEW)
│   ├── 📄 test_api.py              # API endpoint tests ✓ (NEW)
│   ├── 📄 test_audio_processor.py  # Audio tests ✓
│   └── 📄 test_model_detector.py   # Model tests ✓
│
├── 📁 test_samples/
│   └── sample_voice.mp3            # Sample audio file
│
└── 📁 models/                      # Model cache directory
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
# API Settings
API_KEY=your_secret_key
ENVIRONMENT=development

# Server Settings
HOST=0.0.0.0
PORT=8000

# Model Settings
MAX_AUDIO_SIZE_MB=10
```

### Config Options (config.py)

| Setting                    | Default                                  | Description            |
| -------------------------- | ---------------------------------------- | ---------------------- |
| `SAMPLE_RATE`              | 16000                                    | Audio sample rate (Hz) |
| `MAX_AUDIO_LENGTH_SECONDS` | 30                                       | Maximum audio duration |
| `SUPPORTED_LANGUAGES`      | Tamil, English, Hindi, Malayalam, Telugu | Supported languages    |

---

## 🎯 Model Information

### VoiceGUARD (Mrkomiljon/voiceGUARD)

- **Architecture:** Wav2Vec2ForSequenceClassification
- **Training:** Fine-tuned on deepfake audio datasets
- **Sample Rate:** 16000 Hz
- **Input:** Raw audio waveform
- **Output:** Binary classification (bonafide/spoof)
- **Labels:**
  - `0` = HUMAN (bonafide)
  - `1` = AI_GENERATED (spoof)

### Model Download

The model is automatically downloaded from HuggingFace Hub on first run (~360MB). It's cached in `~/.cache/huggingface/` for subsequent runs.

---

## 🐛 Troubleshooting

### Common Issues

**1. Model download fails**

```bash
# Check internet connection
# Try manual download:
python -c "from transformers import Wav2Vec2ForSequenceClassification; Wav2Vec2ForSequenceClassification.from_pretrained('Mrkomiljon/voiceGUARD')"
```

**2. CUDA out of memory**

```python
# Force CPU usage in voiceguard_detector.py
detector = VoiceGUARDDetector(device="cpu")
```

**3. Audio format not supported**

```bash
# Install ffmpeg for MP3 support
# Windows: winget install ffmpeg
# Mac: brew install ffmpeg
# Linux: apt install ffmpeg
```

**4. Audio too short error**

```
# Minimum audio length: 0.5 seconds
# Maximum audio length: 30 seconds
```

---

## 📈 Performance

| Metric               | Value                    |
| -------------------- | ------------------------ |
| Model Size           | ~360MB                   |
| Inference Time (CPU) | 1-3 seconds              |
| Inference Time (GPU) | <0.5 seconds             |
| Sample Rate          | 16000 Hz                 |
| Max Audio Length     | 30 seconds               |
| Supported Formats    | MP3, WAV, FLAC, OGG, M4A |

---

## 📅 Development Progress

### Completion Status

```
[████████████████░░░░] 80% Complete

✓ Feature 1: Project Setup (100%)
✓ Feature 2: Audio Processing (100%)
✓ Feature 3: VoiceGUARD Model Integration (100%) ← CURRENT
⏳ Feature 4: API Authentication (0%)
⏳ Feature 5: Deployment (0%)
⏳ Feature 6: Documentation (50%)
```

### Remaining Tasks

- [ ] Add API key authentication
- [ ] Docker containerization
- [ ] Deploy to Railway/Render
- [ ] Performance optimization
- [ ] Create demo video

---

## 🔧 Dependencies

### Core Dependencies

```
fastapi>=0.104.1          # Web framework
uvicorn>=0.24.0           # ASGI server
librosa>=0.10.1           # Audio processing
torch>=2.1.0              # Deep learning
transformers>=4.35.0      # HuggingFace models
pydantic>=2.5.0           # Data validation
```

See `requirements.txt` for complete list.

---

## 📚 Resources

- [VoiceGUARD Model](https://huggingface.co/Mrkomiljon/voiceGUARD)
- [Wav2Vec2 Documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Librosa Documentation](https://librosa.org/)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- [Mrkomiljon](https://huggingface.co/Mrkomiljon) for the VoiceGUARD model
- HuggingFace for model hosting and transformers library
- FastAPI community
- GUVI for organizing the hackathon

---

**Last Updated:** February 3, 2026  
**Status:** Feature 3 Complete ✓  
**Next:** Feature 4 - API Authentication & Deployment
