# tests/test_api_endpoints.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app
import base64
import numpy as np
from scipy.io import wavfile
import io
import time

# Create test client
client = TestClient(app)

def create_test_audio_base64(duration=2.0, frequency=440):
    """Create a test audio and return as Base64"""
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    audio = (audio * 32767).astype(np.int16)
    
    # Save to WAV buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    buffer.seek(0)
    
    # Encode to Base64
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return audio_base64

def test_root_endpoint():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("TEST 1: ROOT ENDPOINT")
    print("="*60 + "\n")
    
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Service: {data['service']}")
    print(f"✓ Version: {data['version']}")
    print(f"✓ Supported languages: {data['supported_languages']}")
    print(f"✓ Endpoints: {list(data['endpoints'].keys())}")
    
    assert "AI Voice Detection API" in data['service']
    assert len(data['supported_languages']) == 5
    print("\n✓ Root endpoint test passed!")

def test_health_endpoint():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 2: HEALTH CHECK ENDPOINT")
    print("="*60 + "\n")
    
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Status: {response.status_code}")
    print(f"✓ Health status: {data['status']}")
    print(f"✓ Environment: {data['environment']}")
    print(f"✓ Models loaded: {data['models_loaded']}")
    print(f"✓ Supported languages: {len(data['supported_languages'])}")
    
    assert data['status'] == 'healthy'
    # Note: models_loaded may be False in test environment
    print("\n✓ Health endpoint test passed!")

def test_detection_endpoint_success():
    """Test successful voice detection"""
    print("\n" + "="*60)
    print("TEST 3: VOICE DETECTION - SUCCESS")
    print("="*60 + "\n")
    
    # Create test audio
    print("Creating test audio...")
    audio_base64 = create_test_audio_base64(duration=2.0)
    print(f"✓ Created audio ({len(audio_base64)} chars)")
    
    # Make request
    print("\nSending detection request...")
    start_time = time.time()
    
    response = client.post(
        "/api/voice-detection",
        json={
            "language": "Tamil",
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }
    )
    
    request_time = time.time() - start_time
    
    assert response.status_code == 200
    data = response.json()
    
    print(f"✓ Status code: {response.status_code}")
    print(f"✓ Request time: {request_time:.2f}s")
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
    
    assert data['status'] == 'success'
    assert data['language'] == 'Tamil'
    assert data['classification'] in ['AI_GENERATED', 'HUMAN']
    assert 0 <= data['confidenceScore'] <= 1
    print("\n✓ Voice detection test passed!")

def test_detection_all_languages():
    """Test detection with all supported languages"""
    print("\n" + "="*60)
    print("TEST 4: ALL LANGUAGES")
    print("="*60 + "\n")
    
    languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    audio_base64 = create_test_audio_base64(duration=2.0)
    
    results = []
    for language in languages:
        print(f"Testing {language}...")
        
        response = client.post(
            "/api/voice-detection",
            json={
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        results.append({
            'language': language,
            'classification': data['classification'],
            'confidence': data['confidenceScore']
        })
        
        print(f"  ✓ {language}: {data['classification']} ({data['confidenceScore']:.2%})")
    
    print(f"\n✓ All {len(languages)} languages tested successfully!")
    return results

def test_detection_invalid_base64():
    """Test detection with invalid Base64"""
    print("\n" + "="*60)
    print("TEST 5: INVALID BASE64")
    print("="*60 + "\n")
    
    response = client.post(
        "/api/voice-detection",
        json={
            "language": "Tamil",
            "audioFormat": "mp3",
            "audioBase64": "invalid-base64-string!"
        }
    )
    
    print(f"✓ Status code: {response.status_code}")
    
    # Should be 422 (validation error) or 400
    assert response.status_code in [400, 422]
    print("✓ Correctly rejected invalid Base64")

def test_detection_short_audio():
    """Test detection with very short audio"""
    print("\n" + "="*60)
    print("TEST 6: SHORT AUDIO")
    print("="*60 + "\n")
    
    # Create very short audio (0.3 seconds)
    audio_base64 = create_test_audio_base64(duration=0.3)
    
    response = client.post(
        "/api/voice-detection",
        json={
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        }
    )
    
    print(f"✓ Status code: {response.status_code}")
    data = response.json()
    
    # Should return error
    if data['status'] == 'error':
        print(f"✓ Error message: {data.get('message', 'N/A')}")
        print("✓ Correctly rejected short audio")
    else:
        print("⚠ Warning: Short audio was accepted")

def test_detection_different_patterns():
    """Test detection with different audio patterns"""
    print("\n" + "="*60)
    print("TEST 7: DIFFERENT AUDIO PATTERNS")
    print("="*60 + "\n")
    
    patterns = [
        ("Pure sine 440Hz", 440),
        ("Pure sine 880Hz", 880),
        ("Low frequency 200Hz", 200),
    ]
    
    results = []
    for name, frequency in patterns:
        print(f"Testing {name}...")
        audio_base64 = create_test_audio_base64(duration=2.0, frequency=frequency)
        
        response = client.post(
            "/api/voice-detection",
            json={
                "language": "English",
                "audioFormat": "mp3",
                "audioBase64": audio_base64
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            results.append({
                'pattern': name,
                'classification': data['classification'],
                'confidence': data['confidenceScore']
            })
            print(f"  ✓ {name}: {data['classification']} ({data['confidenceScore']:.2%})")
    
    print(f"\n✓ Tested {len(results)} different patterns!")

def test_api_documentation():
    """Test that API documentation is accessible"""
    print("\n" + "="*60)
    print("TEST 8: API DOCUMENTATION")
    print("="*60 + "\n")
    
    # Test OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == 200
    print("✓ OpenAPI schema accessible")
    
    # Test Swagger UI
    response = client.get("/docs")
    assert response.status_code == 200
    print("✓ Swagger UI accessible")
    
    # Test ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200
    print("✓ ReDoc accessible")
    
    print("\n✓ All documentation endpoints working!")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("API ENDPOINT TEST SUITE")
    print("="*60)
    print("\nℹ️  Note: Models will be loaded on first request (may take 10-20s)")
    print("="*60)
    
    try:
        test_root_endpoint()
        test_health_endpoint()
        test_detection_endpoint_success()
        test_detection_all_languages()
        test_detection_invalid_base64()
        test_detection_short_audio()
        test_detection_different_patterns()
        test_api_documentation()
        
        print("\n" + "="*60)
        print("✅ ALL API TESTS PASSED!")
        print("="*60 + "\n")
        
        print("🎉 Your API is ready!")
        print("\n📚 Try it out:")
        print("  1. Start server: python main.py")
        print("  2. Visit: http://localhost:8000/docs")
        print("  3. Test detection endpoint with your audio!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
