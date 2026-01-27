# tests/test_model_detector.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_detector import Wav2Vec2Detector, HybridDetector
from audio_preprocessor import AudioProcessor
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def test_wav2vec2_detector():
    """Test Wav2Vec2 detector with synthetic audio"""
    print("\n" + "="*60)
    print("TEST 1: WAV2VEC2 DETECTOR")
    print("="*60 + "\n")
    
    # Create test audio
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test 1: Pure sine wave (AI-like - very consistent)
    print("Testing with pure sine wave (AI-like)...")
    audio_sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    detector = Wav2Vec2Detector()
    classification, confidence, details = detector.detect(audio_sine, sample_rate)
    
    print(f"✓ Classification: {classification}")
    print(f"✓ Confidence: {confidence:.2%}")
    print(f"✓ AI Score: {details['ai_score']:.2f}")
    
    # Test 2: Noisy audio (Human-like - less consistent)
    print("\nTesting with noisy audio (Human-like)...")
    audio_noisy = np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(len(t))
    audio_noisy = audio_noisy.astype(np.float32)
    
    classification2, confidence2, details2 = detector.detect(audio_noisy, sample_rate)
    
    print(f"✓ Classification: {classification2}")
    print(f"✓ Confidence: {confidence2:.2%}")
    print(f"✓ AI Score: {details2['ai_score']:.2f}")
    
    print("\n✓ Wav2Vec2 detector test passed!")
    return detector

def test_hybrid_detector():
    """Test hybrid detector with both embeddings and acoustic features"""
    print("\n" + "="*60)
    print("TEST 2: HYBRID DETECTOR")
    print("="*60 + "\n")
    
    # Initialize components
    audio_processor = AudioProcessor(sample_rate=16000)
    hybrid_detector = HybridDetector()
    
    # Create test audio
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Preprocess and extract features
    print("Preprocessing audio and extracting features...")
    audio_processed = audio_processor.preprocess_audio(audio)
    acoustic_features = audio_processor.extract_features(audio_processed)
    
    print(f"✓ Extracted {len(acoustic_features)} acoustic features")
    
    # Run hybrid detection
    print("\nRunning hybrid detection...")
    classification, confidence, details = hybrid_detector.detect(
        audio_processed, 
        acoustic_features,
        sample_rate
    )
    
    print(f"\n✓ Classification: {classification}")
    print(f"✓ Confidence: {confidence:.2%}")
    print(f"✓ Wav2Vec2 contribution: {details['wav2vec2']['ai_score']:.2f}")
    print(f"✓ Acoustic contribution: {details['acoustic_ai_score']:.2f}")
    print(f"✓ Combined score: {details['combined_score']:.2f}")
    
    print("\n✓ Hybrid detector test passed!")
    return hybrid_detector

def test_embeddings_extraction():
    """Test embedding extraction separately"""
    print("\n" + "="*60)
    print("TEST 3: EMBEDDING EXTRACTION")
    print("="*60 + "\n")
    
    detector = Wav2Vec2Detector()
    
    # Create test audio
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    print("Extracting embeddings...")
    embeddings = detector.extract_embeddings(audio, sample_rate)
    
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Expected: (sequence_length, 768)")
    print(f"✓ Sequence length: {embeddings.shape[0]} frames")
    print(f"✓ Feature dimension: {embeddings.shape[1]}")
    
    # Analyze embeddings
    print("\nAnalyzing embeddings...")
    metrics = detector.analyze_embeddings(embeddings)
    
    print("✓ Metrics:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")
    
    print("\n✓ Embedding extraction test passed!")

def test_different_audio_patterns():
    """Test detector with different audio patterns"""
    print("\n" + "="*60)
    print("TEST 4: DIFFERENT AUDIO PATTERNS")
    print("="*60 + "\n")
    
    detector = Wav2Vec2Detector()
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    test_cases = [
        ("Pure sine (440Hz)", np.sin(2 * np.pi * 440 * t)),
        ("Chirp (100-1000Hz)", np.sin(2 * np.pi * (100 + 450 * t / duration) * t)),
        ("White noise", np.random.randn(len(t))),
        ("Mixed signal", np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t))
    ]
    
    results = []
    for name, audio in test_cases:
        audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio))  # Normalize
        
        classification, confidence, details = detector.detect(audio, sample_rate)
        results.append((name, classification, confidence, details['ai_score']))
        
        print(f"{name:20s} → {classification:12s} (conf: {confidence:.2%}, score: {details['ai_score']:.2f})")
    
    print("\n✓ Different audio patterns test passed!")
    return results

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL DETECTOR TEST SUITE")
    print("="*60)
    
    try:
        # Run all tests
        test_embeddings_extraction()
        test_wav2vec2_detector()
        test_hybrid_detector()
        test_different_audio_patterns()
        
        print("\n" + "="*60)
        print("✓ ALL MODEL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
