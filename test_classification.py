"""Quick test script for classification"""
from voiceguard_detector import VoiceGUARDDetector
import librosa

detector = VoiceGUARDDetector()
detector.load_model()

# Test with available audio
print("Testing audio sample...")
audio, sr = librosa.load('test_samples/sample_voice.mp3', sr=16000)
result = detector.classify(audio, sr)
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
print(f"Predicted class ID: {result['details']['predicted_class_id']}")
