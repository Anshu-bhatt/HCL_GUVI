# tests/test_api_live.py
"""
Live API test - requires running server
Run: python main.py
Then: python tests/test_api_live.py
"""

import requests
import base64
import numpy as np
from scipy.io import wavfile
import io
import time

API_URL = "http://localhost:8000"

def create_test_audio_base64(duration=2.0, frequency=440):
    """Create a test audio and return as Base64"""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    audio = (audio * 32767).astype(np.int16)
    
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    buffer.seek(0)
    
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return audio_base64

def test_server_running():
    """Check if server is running"""
    print("\n" + "="*60)
    print("TEST 1: SERVER CONNECTION")
    print("="*60 + "\n")
    
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        print(f"✓ Server is running at {API_URL}")
        print(f"✓ Status: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Server not running at {API_URL}")
        print("\n⚠️  Please start the server first:")
        print("   python main.py")
        return False

def test_health_check():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 2: HEALTH CHECK")
    print("="*60 + "\n")
    
    response = requests.get(f"{API_URL}/health")
    data = response.json()
    
    print(f"✓ Status: {data['status']}")
    print(f"✓ Environment: {data['environment']}")
    print(f"✓ Models loaded: {data['models_loaded']}")
    print(f"✓ Languages: {len(data['supported_languages'])}")
    
    if not data['models_loaded']:
        print("\n⚠️  Warning: Models not loaded yet")
        print("   They will load on first detection request")

def test_voice_detection():
    """Test voice detection endpoint"""
    print("\n" + "="*60)
    print("TEST 3: VOICE DETECTION")
    print("="*60 + "\n")
    
    print("Creating test audio...")
    audio_base64 = create_test_audio_base64(duration=2.0)
    print(f"✓ Created audio ({len(audio_base64)} chars)")
    
    print("\nSending detection request...")
    print("⏳ First request may take 10-20s (loading models)...")
    
    start_time = time.time()
    
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        json={
            "language": "Tamil",
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        },
        timeout=60
    )
    
    request_time = time.time() - start_time
    
    print(f"✓ Response received in {request_time:.2f}s")
    print(f"✓ Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n📊 Detection Results:")
        print(f"  - Status: {data['status']}")
        print(f"  - Language: {data['language']}")
        print(f"  - Classification: {data['classification']}")
        print(f"  - Confidence: {data['confidenceScore']:.2%}")
        print(f"  - Explanation: {data['explanation']}")
        
        if 'details' in data:
            print(f"\n📈 Details:")
            for key, value in data['details'].items():
                print(f"  - {key}: {value}")
        
        return True
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_all_languages():
    """Test all supported languages"""
    print("\n" + "="*60)
    print("TEST 4: ALL LANGUAGES")
    print("="*60 + "\n")
    
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    audio_base64 = create_test_audio_base64(duration=2.0)
    
    print(f"Testing {len(languages)} languages...\n")
    
    for i, language in enumerate(languages, 1):
        print(f"[{i}/{len(languages)}] Testing {language}...", end=" ")
        
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json={
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ {data['classification']} ({data['confidenceScore']:.2%})")
        else:
            print(f"❌ Failed ({response.status_code})")
    
    print(f"\n✓ Tested all {len(languages)} languages!")

def test_different_audio():
    """Test with different audio patterns"""
    print("\n" + "="*60)
    print("TEST 5: DIFFERENT AUDIO PATTERNS")
    print("="*60 + "\n")
    
    patterns = [
        ("Pure sine 440Hz", 440),
        ("Pure sine 880Hz", 880),
        ("Low frequency 200Hz", 200),
    ]
    
    for name, frequency in patterns:
        print(f"Testing {name}...", end=" ")
        audio_base64 = create_test_audio_base64(duration=2.0, frequency=frequency)
        
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ {data['classification']} ({data['confidenceScore']:.2%})")
        else:
            print(f"❌ Failed")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 LIVE API TEST SUITE")
    print("="*60)
    print("\nℹ️  This test requires a running server")
    print("   Start with: python main.py")
    print("="*60)
    
    if not test_server_running():
        exit(1)
    
    try:
        test_health_check()
        test_voice_detection()
        test_all_languages()
        test_different_audio()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
        print("🎉 Your API is fully functional!")
        print("\n📚 Next steps:")
        print("  1. Visit: http://localhost:8000/docs")
        print("  2. Try the interactive API documentation")
        print("  3. Test with real audio samples")
        print("  4. Deploy to production!")
        
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
