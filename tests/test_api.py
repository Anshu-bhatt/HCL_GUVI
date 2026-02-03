# tests/test_api.py
"""
Test cases for FastAPI endpoints
"""

import pytest
import base64
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app, detector
from voiceguard_detector import VoiceGUARDDetector


@pytest.fixture(scope="module")
def loaded_detector():
    """Load detector once for all tests"""
    global detector
    import main
    if main.detector is None:
        main.detector = VoiceGUARDDetector()
        main.detector.load_model()
    return main.detector


@pytest.fixture
def client(loaded_detector):
    """Create test client with loaded detector"""
    return TestClient(app)


@pytest.fixture
def sample_audio_base64():
    """Generate sample audio and encode as base64"""
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    # Save to temporary file and encode
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        f.flush()
        
        with open(f.name, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode()
    
    return audio_base64


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_returns_api_info(self, client):
        """Test root endpoint returns API information"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_returns_status(self, client):
        """Test health endpoint returns status"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "device" in data


class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info_returns_details(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_name" in data
        assert "supported_labels" in data
        assert "HUMAN" in data["supported_labels"]
        assert "AI_GENERATED" in data["supported_labels"]


class TestClassifyEndpoint:
    """Test /classify endpoint"""
    
    def test_classify_valid_audio(self, client, sample_audio_base64):
        """Test classification with valid audio"""
        response = client.post(
            "/classify",
            json={"audio_base64": sample_audio_base64}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "classification" in data
        assert data["classification"] in ["AI_GENERATED", "HUMAN"]
        
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        
        assert "confidence_level" in data
        assert "probabilities" in data
        assert "description" in data
        assert "details" in data
    
    def test_classify_with_language_hint(self, client, sample_audio_base64):
        """Test classification with language hint"""
        response = client.post(
            "/classify",
            json={
                "audio_base64": sample_audio_base64,
                "language": "English"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["classification"] in ["AI_GENERATED", "HUMAN"]
    
    def test_classify_invalid_base64(self, client):
        """Test classification with invalid base64"""
        # Invalid base64 that's long enough to pass min_length but is garbage
        invalid_base64 = "a" * 200  # Long enough but not valid audio
        response = client.post(
            "/classify",
            json={"audio_base64": invalid_base64}
        )
        
        # Should fail with either 400 (bad request) or 500 (server error)
        assert response.status_code in [400, 422, 500]
    
    def test_classify_empty_audio(self, client):
        """Test classification with empty base64"""
        response = client.post(
            "/classify",
            json={"audio_base64": ""}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_classify_missing_field(self, client):
        """Test classification with missing audio field"""
        response = client.post(
            "/classify",
            json={}
        )
        
        assert response.status_code == 422  # Validation error


class TestResponseStructure:
    """Test response structure matches API models"""
    
    def test_classify_response_structure(self, client, sample_audio_base64):
        """Test classify response has correct structure"""
        response = client.post(
            "/classify",
            json={"audio_base64": sample_audio_base64}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check top-level fields
        required_fields = [
            "classification",
            "confidence", 
            "confidence_level",
            "probabilities",
            "description",
            "details"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Check probabilities structure
        assert "HUMAN" in data["probabilities"]
        assert "AI_GENERATED" in data["probabilities"]
        
        # Check details structure
        details = data["details"]
        assert "model" in details
        assert "audio_duration_seconds" in details
        assert "sample_rate" in details
        assert "device" in details
        assert "raw_scores" in details


class TestErrorHandling:
    """Test error handling"""
    
    def test_404_for_unknown_endpoint(self, client):
        """Test 404 for unknown endpoint"""
        response = client.get("/unknown/endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method"""
        response = client.get("/classify")  # Should be POST
        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
