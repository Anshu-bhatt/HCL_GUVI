# test_model_weights.py
"""Quick test to verify duration-aware weights are working"""

import numpy as np
from model_detector import HybridDetector

# Create detector
detector = HybridDetector()

# Test with short audio (1.2 seconds)
short_audio = np.random.rand(int(1.2 * 16000)).astype(np.float32)
short_features = {
    'spectral_centroid_std': 50,
    'zcr_std': 0.03,
    'rms_std': 0.15,
    'mfcc_0_std': 5, 'mfcc_1_std': 5, 'mfcc_2_std': 5,
    'mfcc_3_std': 5, 'mfcc_4_std': 5, 'mfcc_5_std': 5,
    'mfcc_6_std': 5, 'mfcc_7_std': 5, 'mfcc_8_std': 5,
    'mfcc_9_std': 5, 'mfcc_10_std': 5, 'mfcc_11_std': 5,
    'mfcc_12_std': 5
}

print("\n" + "="*60)
print("TESTING DURATION-AWARE WEIGHTS")
print("="*60)

print(f"\nShort Audio Test (1.2 seconds):")
result = detector.detect(short_audio, short_features, 16000)
classification, confidence, details = result
print(f"  Classification: {classification}")
print(f"  Confidence: {confidence:.4f}")
print(f"  Is short audio: {details.get('is_short_audio', False)}")
print(f"  Duration: {details.get('audio_duration', 0):.2f}s")
print(f"  Weights used: {details.get('weights_used', {})}")
print(f"  Combined score: {details.get('combined_score', 0):.4f}")

# Test with normal audio (5 seconds)  
normal_audio = np.random.rand(int(5.0 * 16000)).astype(np.float32)
print(f"\nNormal Audio Test (5.0 seconds):")
result2 = detector.detect(normal_audio, short_features, 16000)
classification2, confidence2, details2 = result2
print(f"  Classification: {classification2}")
print(f"  Confidence: {confidence2:.4f}")
print(f"  Is short audio: {details2.get('is_short_audio', False)}")
print(f"  Duration: {details2.get('audio_duration', 0):.2f}s")
print(f"  Weights used: {details2.get('weights_used', {})}")
print(f"  Combined score: {details2.get('combined_score', 0):.4f}")

print("\n" + "="*60)
