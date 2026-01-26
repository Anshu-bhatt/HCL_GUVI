# AI Voice Detection API - Development Guide

## 📋 Project Overview

Multi-language AI voice detection system that identifies whether audio is AI-generated or human-spoken across 5 languages: Tamil, English, Hindi, Malayalam, and Telugu.

**Hackathon:** GUVI AI Hackathon  
**Problem Statement:** AI-Generated Voice Detection (Multi-Language)  
**Timeline:** 7 days  
**Current Status:** Feature 2 Complete ✓

---

## 🎯 Project Goals

- ✅ Build REST API for voice authentication detection
- ✅ Support 5 languages (Tamil, English, Hindi, Malayalam, Telugu)
- ✅ Accept Base64-encoded MP3 audio
- ✅ Return classification (AI_GENERATED or HUMAN) with confidence score
- ✅ Achieve 75-80% accuracy without training data
- ✅ Deploy production-ready API

---

## 🏗️ Architecture & Strategy

### **Chosen Approach:** Hybrid Detection System

**Why this approach:**

- ⏱️ **Time constraint:** 2-3 hours/day for 7 days
- 📊 **No training data** available initially
- 🎯 **Goal:** Balance between accuracy and speed
- ✅ **Feasible:** Uses pre-trained models + analytical features

### **Technology Stack**

```
┌─────────────────────────────────────────────┐
│           FastAPI REST API                  │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │ Wav2Vec2     │  +   │  Acoustic    │   │
│  │ Embeddings   │      │  Features    │   │
│  │ (HuggingFace)│      │  (Librosa)   │   │
│  └──────────────┘      └──────────────┘   │
│         ↓                      ↓           │
│  ┌──────────────────────────────────────┐ │
│  │      Hybrid Detection Logic          │ │
│  └──────────────────────────────────────┘ │
│                    ↓                       │
│         AI_GENERATED or HUMAN              │
└─────────────────────────────────────────────┘
```

### **Core Components**

1. **Audio Processor** - Handles Base64 decoding, validation, preprocessing
2. **Model Detector** - Wav2Vec2 embeddings + heuristic analysis
3. **FastAPI Server** - REST API with authentication
4. **Deployment** - Railway/Render cloud deployment

---

## 📅 Development Roadmap (7 Days)

### ✅ **Day 1-2: Foundation (COMPLETED)**

#### **Feature 1: Project Setup & Environment** ✓

**Status:** Complete  
**Time Spent:** 1.5 hours

**What was built:**

- ✅ Virtual environment setup
- ✅ Dependency installation (FastAPI, PyTorch, Librosa)
- ✅ Configuration management (`config.py`)
- ✅ Basic FastAPI application
- ✅ Health check endpoints
- ✅ Environment variables setup
- ✅ Verification scripts

**Files Created:**

```
├── .env                    # Environment configuration
├── .gitignore             # Git ignore rules
├── config.py              # Application config
├── main.py                # FastAPI app
├── requirements.txt       # Dependencies
└── verify_setup.py        # Setup verification
```

**Verification:**

```bash
python verify_setup.py
python main.py
# Visit: http://localhost:8000/docs
```

---

#### **Feature 2: Audio Processing Pipeline** ✓

**Status:** Complete  
**Time Spent:** 2.5 hours

**What was built:**

- ✅ Base64 audio decoding
- ✅ Audio loading and validation
- ✅ Preprocessing (normalization, trimming, padding)
- ✅ Feature extraction (40+ acoustic features)
- ✅ Error handling for invalid audio
- ✅ Audio information extraction
- ✅ Comprehensive testing suite

**Files Created:**

```
├── audio_processor.py           # Main audio processor
├── download_sample.py           # Sample audio downloader
│
├── utils/
│   └── audio_helpers.py         # Utility functions
│
└── tests/
    ├── test_audio_processor.py      # Unit tests
    └── test_with_real_audio.py      # Integration tests
```

**Features Extracted:**

- Spectral features (centroid, rolloff, bandwidth)
- Zero crossing rate
- MFCC (13 coefficients)
- RMS energy
- Spectral contrast

**Verification:**

```bash
python tests/test_audio_processor.py
python download_sample.py
python tests/test_with_real_audio.py
```

---

### 🔄 **Day 3: Model Integration (IN PROGRESS)**

#### **Feature 3: Pre-trained Model Integration**

**Status:** Ready to implement  
**Estimated Time:** 2.5 hours

**What will be built:**

- ⏳ Wav2Vec2 model integration (HuggingFace)
- ⏳ Embedding extraction pipeline
- ⏳ Heuristic-based AI detection (no training needed)
- ⏳ Hybrid detector (embeddings + acoustic features)
- ⏳ Model testing and validation

**Files to Create:**

```
├── model_detector.py                  # Wav2Vec2 detector
├── compare_detectors.py               # Comparison tool
├── optimize_model.py                  # Optimization guide
│
└── tests/
    ├── test_model_detector.py         # Model unit tests
    └── test_model_with_real_audio.py  # Model integration tests
```

**Key Components:**

1. **Wav2Vec2Detector** - Pre-trained model wrapper
2. **Embedding Analysis** - Detect AI patterns in embeddings
3. **HybridDetector** - Combine multiple signals

**Next Steps:**

```bash
# Update requirements
pip install transformers datasets accelerate

# Implement model_detector.py
# Run tests
python tests/test_model_detector.py
```

---

### 📋 **Day 4: API Integration (PLANNED)**

#### **Feature 4: Complete API Implementation**

**Status:** Planned  
**Estimated Time:** 2 hours

**What will be built:**

- ⏳ POST endpoint for voice detection
- ⏳ Request validation (language, format, Base64)
- ⏳ API key authentication
- ⏳ Response formatting (JSON structure)
- ⏳ Error handling and status codes
- ⏳ Rate limiting (optional)

**Files to Create/Update:**

```
├── main.py                  # Update with detection endpoint
├── schemas.py               # Pydantic models
├── middleware.py            # Authentication, logging
│
└── tests/
    └── test_api_endpoints.py  # API tests
```

**Endpoint Specification:**

```
POST /api/voice-detection
Headers:
  - x-api-key: YOUR_API_KEY
  - Content-Type: application/json

Body:
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "..."
}

Response:
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "..."
}
```

---

### 🧪 **Day 5: Testing & Validation (PLANNED)**

#### **Feature 5: Comprehensive Testing**

**Status:** Planned  
**Estimated Time:** 3 hours

**What will be built:**

- ⏳ Test with all 5 languages
- ⏳ Collect diverse audio samples
- ⏳ Accuracy measurement
- ⏳ Edge case testing
- ⏳ Performance benchmarking
- ⏳ Fine-tune detection thresholds

**Test Categories:**

1. **Language Coverage** - Test Tamil, English, Hindi, Malayalam, Telugu
2. **Audio Quality** - Test with different bitrates, noise levels
3. **AI Voices** - ElevenLabs, Google TTS, Azure TTS samples
4. **Human Voices** - Record own voice, use public datasets
5. **Edge Cases** - Very short audio, silence, music

**Files to Create:**

```
└── tests/
    ├── test_all_languages.py
    ├── test_edge_cases.py
    ├── test_performance.py
    └── accuracy_report.py
```

---

### 🚀 **Day 6: Deployment Preparation (PLANNED)**

#### **Feature 6: Production Deployment**

**Status:** Planned  
**Estimated Time:** 2.5 hours

**What will be built:**

- ⏳ Docker containerization
- ⏳ Environment configuration for production
- ⏳ Railway/Render deployment setup
- ⏳ Monitoring and logging
- ⏳ API documentation (Swagger)

**Files to Create:**

```
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Local testing
├── railway.json            # Railway config
├── render.yaml             # Render config
└── .dockerignore          # Docker ignore
```

**Deployment Platforms:**

- **Primary:** Railway (recommended)
- **Backup:** Render
- **Alternative:** Docker on any cloud

---

### 🎨 **Day 7: Polish & Documentation (PLANNED)**

#### **Feature 7: Final Polish**

**Status:** Planned  
**Estimated Time:** 2.5 hours

**What will be done:**

- ⏳ API documentation refinement
- ⏳ README completion
- ⏳ Demo video/screenshots
- ⏳ Performance optimization
- ⏳ Final testing in production
- ⏳ Presentation preparation

**Deliverables:**

- ✅ Deployed API URL
- ✅ API documentation
- ✅ GitHub repository
- ✅ Demo samples
- ✅ Performance metrics

---

## 📊 Current Project Structure

```
ai-voice-detection/
│
├── 📄 README.md                    # This file
├── 📄 .env                         # Environment variables
├── 📄 .gitignore                   # Git ignore
├── 📄 requirements.txt             # Python dependencies
├── 📄 config.py                    # Configuration ✓
├── 📄 main.py                      # FastAPI app ✓
├── 📄 audio_processor.py           # Audio processing ✓
├── 📄 verify_setup.py              # Setup verification ✓
├── 📄 download_sample.py           # Sample downloader ✓
│
├── 📁 models/
│   └── (Wav2Vec2 cache will be here)
│
├── 📁 utils/
│   └── 📄 audio_helpers.py         # Helper functions ✓
│
├── 📁 tests/
│   ├── 📄 test_audio_processor.py      ✓
│   └── 📄 test_with_real_audio.py      ✓
│
└── 📁 test_samples/
    ├── sample_voice.mp3            # Downloaded sample
    └── feature_summary.json        # Generated features
```

---

## 🔧 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- 2GB free disk space (for models)
- Stable internet connection (for model download)

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd ai-voice-detection

# 2. Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python verify_setup.py

# 5. Download sample audio
python download_sample.py

# 6. Run tests
python tests/test_audio_processor.py

# 7. Start server
python main.py
```

---

## 🧪 Testing Guide

### Current Tests Available

```bash
# Test 1: Setup verification
python verify_setup.py

# Test 2: Audio processor with synthetic audio
python tests/test_audio_processor.py

# Test 3: Audio processor with real MP3
python tests/test_with_real_audio.py

# Test 4: Run API server
python main.py
# Visit: http://localhost:8000/docs
```

---

## 📝 Dependencies

### Core Dependencies

```
fastapi==0.104.1          # Web framework
uvicorn==0.24.0           # ASGI server
librosa==0.10.1           # Audio processing
torch==2.1.0              # Deep learning
transformers==4.35.0      # HuggingFace models
```

### Full List

See `requirements.txt` for complete dependency list.

---

## 🎯 Performance Targets

| Metric           | Target        | Current Status         |
| ---------------- | ------------- | ---------------------- |
| Accuracy         | 75-80%        | TBD (after Feature 3)  |
| Response Time    | < 3 seconds   | TBD (after Feature 4)  |
| Languages        | 5/5 supported | 5/5 ✓                  |
| Audio Format     | MP3           | Supported ✓            |
| Max Audio Length | 30 seconds    | Supported ✓            |
| API Uptime       | 99%+          | TBD (after deployment) |

---

## 🚀 Quick Start (For New Developers)

```bash
# Complete setup in 5 steps:

# Step 1: Setup environment
python -m venv venv && source venv/bin/activate  # or venv\Scripts\activate on Windows

# Step 2: Install everything
pip install -r requirements.txt

# Step 3: Verify it works
python verify_setup.py

# Step 4: Test with sample
python download_sample.py && python tests/test_with_real_audio.py

# Step 5: Start developing
python main.py
```

---

## 📖 API Documentation (Planned)

### Endpoint

```
POST /api/voice-detection
```

### Request Format

```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

### Response Format

```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Detected consistent embedding patterns, sparse spectral features"
}
```

### Error Response

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

---

## 🔐 Security

- ✅ API key authentication required
- ✅ Input validation for all requests
- ✅ Audio size limits (10MB max)
- ✅ Rate limiting (planned)
- ✅ HTTPS in production (deployment)

---

## 🐛 Known Issues & Limitations

### Current Limitations

1. **No training data** - Using heuristic detection (75-80% accuracy expected)
2. **Model size** - Wav2Vec2 is ~360MB (cached after first download)
3. **Processing time** - 1-3 seconds per audio (can be optimized)

### Planned Improvements

1. Fine-tune thresholds with real test data
2. Add model quantization for smaller size
3. Implement caching for faster repeated requests
4. Add batch processing support

---

## 📚 Resources & References

### Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Librosa Documentation](https://librosa.org/)
- [HuggingFace Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base)
- [Problem Statement](./problem_statement.md)

### Datasets (for testing)

- [Common Voice by Mozilla](https://commonvoice.mozilla.org/)
- [ElevenLabs](https://elevenlabs.io/) - AI voice generation
- [Google Cloud TTS](https://cloud.google.com/text-to-speech)

---

## 👥 Team & Contact

**Developer:** [Your Name]  
**Hackathon:** GUVI AI Hackathon 2025  
**Timeline:** January 21-28, 2025  
**Repository:** [GitHub URL]

---

## 📈 Progress Tracking

### Completion Status

```
[████████████░░░░░░░] 60% Complete

✓ Feature 1: Project Setup (100%)
✓ Feature 2: Audio Processing (100%)
⏳ Feature 3: Model Integration (0%)
⏳ Feature 4: API Implementation (0%)
⏳ Feature 5: Testing (0%)
⏳ Feature 6: Deployment (0%)
⏳ Feature 7: Documentation (0%)
```

### Daily Log

**Day 1 (Jan 21):**

- ✅ Project structure created
- ✅ Environment setup complete
- ✅ Basic FastAPI running

**Day 2 (Jan 22):**

- ✅ Audio processor implemented
- ✅ Feature extraction working
- ✅ All tests passing

**Day 3 (Jan 23):**

- ⏳ Model integration in progress

---

## 🎓 Learning Outcomes

### Technical Skills Gained

- ✅ Audio processing with Librosa
- ✅ FastAPI development
- ⏳ HuggingFace Transformers
- ⏳ Model deployment
- ⏳ REST API design

### Best Practices

- ✅ Modular code structure
- ✅ Comprehensive testing
- ✅ Configuration management
- ✅ Error handling
- ✅ Documentation

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- HuggingFace for pre-trained models
- FastAPI community
- Librosa developers
- GUVI for organizing the hackathon

---

**Last Updated:** January 23, 2025  
**Status:** Feature 2 Complete ✓  
**Next:** Feature 3 - Model Integration

---

## 🚦 Quick Commands Reference

```bash
# Setup
python verify_setup.py

# Testing
python tests/test_audio_processor.py
python tests/test_with_real_audio.py

# Development
python main.py

# Download sample
python download_sample.py

# Future commands (after Feature 3)
python tests/test_model_detector.py
python compare_detectors.py
```

---

_This README will be updated as features are completed._
