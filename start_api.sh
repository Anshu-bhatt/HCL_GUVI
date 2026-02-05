#!/bin/bash
# start_api.sh
# Quick start script for VoiceGUARD API

echo "🚀 Starting VoiceGUARD API for Hackathon Evaluation"
echo "=" * 60

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Set production environment
echo "⚙️ Setting up production environment..."
cp .env.production .env

# Test the setup
echo "🔍 Testing API setup..."
python test_production_api.py

echo ""
echo "✅ Setup complete!"
echo "🌐 Starting API server..."
echo "📍 URL: http://localhost:8000/api/v1/detect"
echo "🔑 API Key: GUVI_HACKATHON_VOICEGUARD_2026_AUTH_KEY_123456"
echo "📖 Docs: http://localhost:8000/docs"
echo ""

# Start the server
python deploy.py