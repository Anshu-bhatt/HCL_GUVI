# DEPLOYMENT_GUIDE.md

# 🚀 VoiceGUARD API Deployment Guide

## ⚡ QUICK FREE DEPLOYMENT FOR HACKATHON

### **🎯 1-Click Deploy Options (Recommended)**

| Platform        | Free Tier                | URL Format               | Setup Time |
| --------------- | ------------------------ | ------------------------ | ---------- |
| **Railway** ⭐  | 500 hrs/month, 1GB RAM   | `yourapp.up.railway.app` | 5 minutes  |
| **Render** ⭐   | 750 hrs/month, 0.5GB RAM | `yourapp.onrender.com`   | 10 minutes |
| **Fly.io**      | 2,340 hrs/month, 256MB   | `yourapp.fly.dev`        | 15 minutes |
| **HuggingFace** | Unlimited, ML-optimized  | `username-app.hf.space`  | 20 minutes |

### **🚀 FASTEST DEPLOYMENT: Railway (5 minutes)**

1. **Go to [railway.app](https://railway.app)**
2. **Sign up with GitHub**
3. **Click "New Project" → "Deploy from GitHub repo"**
4. **Select your HCL_GUVI repository**
5. **Add these environment variables:**
   ```
   API_KEY=GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456
   PORT=8000
   ENVIRONMENT=production
   ```
6. **Your API will be live at:** `https://hcl-guvi-production.up.railway.app`

**✅ Done! Your endpoint is ready for hackathon submission.**

---

## Quick Deployment Options

### Option 1: Local Development Server

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set production environment
cp .env.production .env

# 3. Start server
python deploy.py
```

### Option 2: Docker Deployment (Recommended)

```bash
# 1. Build and run with Docker Compose
docker-compose up -d

# 2. Check logs
docker-compose logs -f
```

### Option 3: Cloud Deployment (FREE HOSTING OPTIONS)

## 🆓 FREE DEPLOYMENT PLATFORMS

### 1. **Railway (Recommended - Easiest)**

- **Free Tier:** 500 hours/month, 1GB RAM
- **URL:** `https://your-app-name.up.railway.app`
- **Pros:** Easy setup, good for Python ML apps, persistent storage

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway init
railway deploy

# 3. Set environment variables in Railway dashboard
# API_KEY=GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456
```

### 2. **Render.com**

- **Free Tier:** 750 hours/month, 0.5GB RAM
- **URL:** `https://your-app-name.onrender.com`
- **Pros:** Automatic deployments, good Python support

```bash
# 1. Connect your GitHub repo to Render
# 2. Create new Web Service
# 3. Build Command: pip install -r requirements.txt
# 4. Start Command: python deploy.py
# 5. Add environment variables in dashboard
```

### 3. **Fly.io**

- **Free Tier:** 2,340 hours/month, 256MB RAM
- **URL:** `https://your-app-name.fly.dev`
- **Pros:** Global edge deployment, Docker support

```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Login and launch
fly auth login
fly launch

# 3. Deploy
fly deploy
```

### 4. **Hugging Face Spaces (ML-Optimized)**

- **Free Tier:** Unlimited, 16GB storage
- **URL:** `https://your-username-voiceguard.hf.space`
- **Pros:** Perfect for ML models, automatic model loading

```python
# Create app.py for Hugging Face Spaces
import gradio as gr
from deploy import app
import uvicorn

# Gradio interface wrapper
def classify_audio(audio_file):
    # Your classification logic here
    return {"classification": "AI_GENERATED", "confidence": 0.95}

iface = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.JSON(),
    title="VoiceGUARD API"
)

if __name__ == "__main__":
    iface.launch()
```

### 5. **PythonAnywhere**

- **Free Tier:** 1 web app, 3 months
- **URL:** `https://your-username.pythonanywhere.com`
- **Pros:** Python-focused, easy setup

### 6. **Heroku (Limited Free)**

- **Free Tier:** 550 hours/month (sleeps after 30min idle)
- **URL:** `https://your-app-name.herokuapp.com`

```bash
# 1. Create Heroku app
heroku create your-voiceguard-api

# 2. Add Procfile
echo "web: python deploy.py" > Procfile

# 3. Set environment variables
heroku config:set API_KEY=GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456
heroku config:set PORT=8000

# 4. Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### 7. **Streamlit Cloud**

- **Free Tier:** Unlimited public apps
- **URL:** `https://share.streamlit.io/your-username/repo-name`
- **Pros:** Great for ML demos, easy integration

### 8. **Google Cloud Run (Free Tier)**

- **Free Tier:** 2 million requests/month
- **URL:** `https://your-service-name-xyz-uc.a.run.app`

```bash
# 1. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/voiceguard-api

# 2. Deploy to Cloud Run
gcloud run deploy --image gcr.io/PROJECT_ID/voiceguard-api --platform managed
```

## 🎯 RECOMMENDED FOR HACKATHON: Railway or Render

### **Railway Setup (5 minutes):**

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Add environment variables:
   - `API_KEY=GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456`
   - `PORT=8000`
6. Deploy automatically gets a URL like: `https://voiceguard-production.up.railway.app`

### **Render Setup (10 minutes):**

1. Go to [render.com](https://render.com)
2. Connect GitHub repository
3. Create "Web Service"
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `python deploy.py`
6. Add environment variables in dashboard
7. Deploy gets URL like: `https://voiceguard-api.onrender.com`

## Configuration

### Environment Variables

```bash
API_KEY=GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
MAX_AUDIO_SIZE_MB=10
MAX_AUDIO_LENGTH_SECONDS=30
```

### Required Dependencies

See `requirements.txt` for full list:

- fastapi
- uvicorn
- transformers
- torch
- librosa
- soundfile
- numpy

## Testing Deployment

### 1. Test locally

```bash
python test_production_api.py
```

### 2. Test remotely

```bash
# Replace YOUR_SERVER with actual URL
curl -X POST "http://YOUR_SERVER:8000/api/v1/detect" \
  -H "X-API-Key: GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456" \
  -F "file=@test_samples/sample_human.wav"
```

## API Documentation

Once deployed, access interactive documentation at:

- **Swagger UI:** `http://YOUR_SERVER:8000/docs`
- **ReDoc:** `http://YOUR_SERVER:8000/redoc`

## Monitoring

### Health Checks

- **Health endpoint:** `/health`
- **API info:** `/api/v1/info`
- **Model status:** Available in health response

### Logs

- Application logs show classification results
- Model loading status
- Authentication attempts
- Error details

## Security

### API Key

- Required for all classification endpoints
- Passed via `X-API-Key` header
- Key: `GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456`

### CORS

- Enabled for all origins (suitable for evaluation)
- Adjust in production as needed

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Ensure internet connection for downloading VoiceGUARD model
   - Check available disk space
   - Verify transformers library version

2. **Audio processing errors**
   - Check supported formats (MP3, WAV, FLAC, OGG, M4A)
   - Verify file size limits
   - Ensure audio duration under 30 seconds

3. **Authentication errors**
   - Verify API key in request headers
   - Check exact header name: `X-API-Key`

### Debug Mode

```bash
# Run with debug logging
ENVIRONMENT=development python deploy.py
```

## Performance

### Model Loading

- First request may take 30-60 seconds (model download)
- Subsequent requests: ~1-3 seconds
- Keep-alive recommended for production

### Resource Requirements

- **RAM:** 2GB minimum, 4GB recommended
- **CPU:** 2 cores minimum
- **Storage:** 2GB for model files
- **GPU:** Optional, improves speed

## Production Checklist

- ✅ Environment variables set
- ✅ API key configured
- ✅ Health checks enabled
- ✅ Error logging configured
- ✅ CORS properly configured
- ✅ Model auto-download working
- ✅ File upload limits set
- ✅ Authentication working

## Support

For deployment issues:

1. Check logs for error details
2. Verify all dependencies installed
3. Test with provided test script
4. Ensure model downloads correctly

**Your API is now ready for hackathon evaluation! 🎉**
