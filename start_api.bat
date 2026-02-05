@echo off
REM start_api.bat
REM Quick start script for VoiceGUARD API (Windows)

echo 🚀 Starting VoiceGUARD API for Hackathon Evaluation
echo ============================================================

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Set production environment
echo ⚙️ Setting up production environment...
copy .env.production .env

REM Test the setup
echo 🔍 Testing API setup...
python test_production_api.py

echo.
echo ✅ Setup complete!
echo 🌐 Starting API server...
echo 📍 URL: http://localhost:8000/api/v1/detect
echo 🔑 API Key: GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456
echo 📖 Docs: http://localhost:8000/docs
echo.

REM Start the server
python deploy.py

pause