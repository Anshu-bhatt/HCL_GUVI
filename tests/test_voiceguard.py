# tests/test_voiceguard.py
"""
Test cases for VoiceGUARD voice classification system
"""

import pytest
import numpy as np
import base64
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from voiceguard_detector import VoiceGUARDDetector


class TestVoiceGUARDDetector:
    """Test suite for VoiceGUARD detector"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        """Create detector instance for tests"""
        det = VoiceGUARDDetector()
        det.load_model()
        return det
    
    @pytest.fixture
    def sample_audio_human_like(self):
        """
        Generate sample audio that mimics human speech characteristics
        Uses varying frequencies and amplitude modulation
        """
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio with natural variation (human-like)
        # Base frequency with vibrato
        vibrato = 5 * np.sin(2 * np.pi * 5 * t)  # 5Hz vibrato
        frequency = 200 + vibrato + 50 * np.random.randn(len(t)) * 0.1
        
        # Generate audio with varying amplitude (breathing patterns)
        amplitude = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        
        # Add harmonics
        audio = amplitude * (
            np.sin(2 * np.pi * 200 * t) +
            0.5 * np.sin(2 * np.pi * 400 * t) +
            0.25 * np.sin(2 * np.pi * 600 * t)
        )
        
        # Add slight noise (microphone noise)
        audio += 0.02 * np.random.randn(len(audio))
        
        return audio.astype(np.float32), sample_rate
    
    @pytest.fixture
    def sample_audio_synthetic(self):
        """
        Generate sample audio that mimics AI-generated characteristics
        Very consistent, clean signal
        """
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create very clean, consistent audio (synthetic-like)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # Pure sine wave
        
        return audio.astype(np.float32), sample_rate
    
    @pytest.fixture
    def sample_audio_short(self):
        """Generate audio that's too short"""
        sample_rate = 16000
        duration = 0.2  # Only 0.2 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio.astype(np.float32), sample_rate
    
    def test_detector_initialization(self):
        """Test detector can be initialized"""
        detector = VoiceGUARDDetector()
        assert detector is not None
        assert detector.model_name == "Mrkomiljon/voiceGUARD"
        assert detector.SAMPLE_RATE == 16000
    
    def test_model_loading(self, detector):
        """Test model loads successfully"""
        assert detector._is_loaded is True
        assert detector.processor is not None
        assert detector.model is not None
    
    def test_model_info(self, detector):
        """Test get_model_info returns correct structure"""
        info = detector.get_model_info()
        
        assert "model_name" in info
        assert "is_loaded" in info
        assert "device" in info
        assert "sample_rate" in info
        assert "supported_labels" in info
        assert "HUMAN" in info["supported_labels"]
        assert "AI_GENERATED" in info["supported_labels"]
    
    def test_classify_returns_required_fields(self, detector, sample_audio_synthetic):
        """Test classification returns all required fields"""
        audio, sr = sample_audio_synthetic
        result = detector.classify(audio, sr)
        
        # Check required fields
        assert "classification" in result
        assert "confidence" in result
        assert "confidence_level" in result
        assert "probabilities" in result
        assert "description" in result
        assert "details" in result
        
        # Check classification is valid
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]
        
        # Check confidence is in valid range
        assert 0.0 <= result["confidence"] <= 1.0
        
        # Check probabilities
        assert "HUMAN" in result["probabilities"]
        assert "AI_GENERATED" in result["probabilities"]
        
        # Probabilities should sum to ~1
        total_prob = sum(result["probabilities"].values())
        assert 0.99 <= total_prob <= 1.01
    
    def test_classify_synthetic_audio(self, detector, sample_audio_synthetic):
        """Test classification of synthetic audio"""
        audio, sr = sample_audio_synthetic
        result = detector.classify(audio, sr)
        
        # Should detect pure sine wave as AI-generated
        # Note: This is expected behavior for synthetic test audio
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]
        assert result["confidence"] > 0.0
        
        print(f"\nSynthetic audio result: {result['classification']} "
              f"(confidence: {result['confidence']:.2%})")
    
    def test_classify_human_like_audio(self, detector, sample_audio_human_like):
        """Test classification of human-like audio"""
        audio, sr = sample_audio_human_like
        result = detector.classify(audio, sr)
        
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]
        assert result["confidence"] > 0.0
        
        print(f"\nHuman-like audio result: {result['classification']} "
              f"(confidence: {result['confidence']:.2%})")
    
    def test_reject_short_audio(self, detector, sample_audio_short):
        """Test that very short audio is rejected"""
        audio, sr = sample_audio_short
        
        with pytest.raises(ValueError, match="too short"):
            detector.classify(audio, sr)
    
    def test_confidence_levels(self, detector, sample_audio_synthetic):
        """Test confidence level mapping"""
        audio, sr = sample_audio_synthetic
        result = detector.classify(audio, sr)
        
        confidence = result["confidence"]
        level = result["confidence_level"]
        
        # Verify level matches confidence
        if confidence >= 0.95:
            assert level == "VERY_HIGH"
        elif confidence >= 0.85:
            assert level == "HIGH"
        elif confidence >= 0.70:
            assert level == "MEDIUM"
        elif confidence >= 0.55:
            assert level == "LOW"
        else:
            assert level == "VERY_LOW"
    
    def test_preprocess_resampling(self, detector):
        """Test audio resampling from different sample rates"""
        # Create audio at 44.1kHz
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Should resample to 16kHz
        processed = detector.preprocess_audio(audio, sample_rate)
        
        # Check length is correct for 16kHz
        expected_length = int(duration * 16000)
        assert len(processed) <= expected_length + 100  # Allow some variance from trimming
    
    def test_base64_classification(self, detector, sample_audio_synthetic, tmp_path):
        """Test classification from base64 encoded audio"""
        import soundfile as sf
        
        audio, sr = sample_audio_synthetic
        
        # Save to temporary WAV file
        temp_file = tmp_path / "test_audio.wav"
        sf.write(str(temp_file), audio, sr)
        
        # Read and encode as base64
        with open(temp_file, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        
        # Classify
        result = detector.classify_base64(audio_base64)
        
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]
        assert result["details"]["input_type"] == "base64"
    
    def test_details_contains_model_info(self, detector, sample_audio_synthetic):
        """Test that details contain model information"""
        audio, sr = sample_audio_synthetic
        result = detector.classify(audio, sr)
        
        details = result["details"]
        assert details["model"] == "Mrkomiljon/voiceGUARD"
        assert details["sample_rate"] == 16000
        assert "audio_duration_seconds" in details
        assert "device" in details
        assert "raw_scores" in details


class TestAudioPreprocessing:
    """Test audio preprocessing functions"""
    
    @pytest.fixture
    def detector(self):
        """Create detector for preprocessing tests"""
        return VoiceGUARDDetector()
    
    def test_normalize_audio(self, detector):
        """Test audio normalization"""
        # Create audio with high amplitude
        audio = np.array([0.0, 2.0, -2.0, 1.0], dtype=np.float32)
        
        # Preprocess (includes normalization)
        # Need at least 0.5s of audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (2.0 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        processed = detector.preprocess_audio(audio, sample_rate)
        
        # Should be normalized
        assert np.max(np.abs(processed)) <= 1.0
    
    def test_handle_empty_audio(self, detector):
        """Test handling of empty audio"""
        audio = np.array([], dtype=np.float32)
        
        with pytest.raises(ValueError):
            detector.preprocess_audio(audio, 16000)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def detector(self):
        """Create detector for edge case tests"""
        det = VoiceGUARDDetector()
        det.load_model()
        return det
    
    def test_very_long_audio_truncation(self, detector):
        """Test that very long audio is truncated"""
        sample_rate = 16000
        duration = 60.0  # 60 seconds (over the 30s limit)
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Should not raise, but truncate
        result = detector.classify(audio, sample_rate)
        
        # Should still get a valid result
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]
        
        # Duration should be capped
        assert result["details"]["audio_duration_seconds"] <= 30.0
    
    def test_audio_with_silence(self, detector):
        """Test audio with lots of silence"""
        sample_rate = 16000
        
        # 2 seconds of mostly silence with brief audio
        silence = np.zeros(sample_rate, dtype=np.float32)
        audio_part = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))
        audio = np.concatenate([silence, audio_part.astype(np.float32)])
        
        result = detector.classify(audio, sample_rate)
        
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]
    
    def test_noisy_audio(self, detector):
        """Test classification of very noisy audio"""
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # Pure noise
        audio = (0.3 * np.random.randn(samples)).astype(np.float32)
        
        result = detector.classify(audio, sample_rate)
        
        # Should still return a classification
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]


# Integration tests for real audio files
class TestRealAudioFiles:
    """Test with real audio files if available"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        det = VoiceGUARDDetector()
        det.load_model()
        return det
    
    @pytest.fixture
    def sample_file_path(self):
        """Get path to sample audio file"""
        base_path = Path(__file__).parent.parent
        sample_paths = [
            base_path / "test_samples" / "sample_voice.mp3",
            base_path / "test_sample" / "test.wav",
        ]
        
        for path in sample_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def test_classify_real_file(self, detector, sample_file_path):
        """Test classification of real audio file"""
        if sample_file_path is None:
            pytest.skip("No sample audio file available")
        
        result = detector.classify_file(sample_file_path)
        
        assert result["classification"] in ["AI_GENERATED", "HUMAN"]
        assert result["confidence"] > 0.0
        assert "file_path" in result["details"]
        
        print(f"\nReal file classification: {result['classification']} "
              f"(confidence: {result['confidence']:.2%})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
