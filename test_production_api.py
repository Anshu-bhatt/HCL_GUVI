# test_production_api.py
"""
Test script for production API endpoints
Verify API is ready for hackathon evaluation
"""

import requests
import json
import base64
import numpy as np
import soundfile as sf
import tempfile
import os
from config import config

# API Configuration
API_BASE_URL = f"http://localhost:{config.PORT}"
API_KEY = config.API_KEY

def create_test_audio():
    """Create a simple test audio file"""
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        return f.name

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"✅ Health Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health test failed: {e}")
        return False

def test_api_info():
    """Test API info endpoint"""
    print("\n🔍 Testing API info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/info")
        print(f"✅ API Info Status: {response.status_code}")
        data = response.json()
        print(f"   API Name: {data.get('api_name')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Problem: {data.get('problem_statement')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ API info test failed: {e}")
        return False

def test_detect_without_auth():
    """Test detect endpoint without authentication"""
    print("\n🔍 Testing authentication requirement...")
    try:
        audio_file = create_test_audio()
        with open(audio_file, 'rb') as f:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/detect",
                files={"file": f}
            )
        print(f"✅ Auth test (should be 401): {response.status_code}")
        os.unlink(audio_file)
        return response.status_code == 401
    except Exception as e:
        print(f"❌ Auth test failed: {e}")
        return False

def test_detect_with_auth():
    """Test detect endpoint with authentication"""
    print("\n🔍 Testing detect endpoint with authentication...")
    try:
        audio_file = create_test_audio()
        with open(audio_file, 'rb') as f:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/detect",
                headers={"X-API-Key": API_KEY},
                files={"file": f}
            )
        
        print(f"✅ Detect Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Classification: {data.get('classification')}")
            print(f"   Confidence: {data.get('confidence')}")
        else:
            print(f"   Error: {response.text}")
        
        os.unlink(audio_file)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Detect test failed: {e}")
        return False

def test_classify_with_auth():
    """Test classify endpoint with Base64 audio"""
    print("\n🔍 Testing classify endpoint with Base64...")
    try:
        # Create test audio
        audio_file = create_test_audio()
        
        # Encode to base64
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Test API
        response = requests.post(
            f"{API_BASE_URL}/api/v1/classify",
            headers={
                "X-API-Key": API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "audio_base64": audio_b64,
                "language": "English"
            }
        )
        
        print(f"✅ Classify Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Classification: {data.get('classification')}")
            print(f"   Confidence: {data.get('confidence')}")
        else:
            print(f"   Error: {response.text}")
        
        os.unlink(audio_file)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Classify test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing VoiceGUARD Production API")
    print("=" * 50)
    
    tests = [
        test_health,
        test_api_info,
        test_detect_without_auth,
        test_detect_with_auth,
        test_classify_with_auth
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 API is ready for hackathon evaluation!")
        print(f"\n📍 API Endpoint: {API_BASE_URL}/api/v1/detect")
        print(f"🔑 API Key: {API_KEY}")
        print(f"📖 Documentation: {API_BASE_URL}/docs")
    else:
        print("\n❌ Some tests failed. Check the API setup.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()