# demo_classify.py
"""
Demo script for VoiceGUARD classification
Test the API with sample audio files
"""

import requests
import base64
import json
import sys
import os
from pathlib import Path


def classify_audio_file(file_path: str, api_url: str = "http://localhost:8000") -> dict:
    """
    Classify an audio file using the VoiceGUARD API
    
    Args:
        file_path: Path to audio file
        api_url: Base URL of the API
        
    Returns:
        Classification result
    """
    # Read and encode audio
    with open(file_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    
    # Make API request
    response = requests.post(
        f"{api_url}/classify",
        json={"audio_base64": audio_base64}
    )
    
    return response.json()


def classify_with_detector(file_path: str) -> dict:
    """
    Classify directly using the VoiceGUARD detector (no API server needed)
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Classification result
    """
    from voiceguard_detector import VoiceGUARDDetector
    
    detector = VoiceGUARDDetector()
    detector.load_model()
    
    return detector.classify_file(file_path)


def print_result(result: dict, file_name: str = "audio") -> None:
    """Pretty print classification result"""
    print("\n" + "=" * 60)
    print(f"CLASSIFICATION RESULT: {file_name}")
    print("=" * 60)
    
    if "error" in result:
        print(f"\n❌ Error: {result.get('detail', result.get('message', 'Unknown error'))}")
        return
    
    classification = result.get("classification", "Unknown")
    confidence = result.get("confidence", 0)
    confidence_level = result.get("confidence_level", "Unknown")
    
    # Emoji based on result
    emoji = "🤖" if classification == "AI_GENERATED" else "👤"
    
    print(f"\n{emoji} Classification: {classification}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"📈 Confidence Level: {confidence_level}")
    
    # Probabilities
    probs = result.get("probabilities", {})
    print("\n--- Probabilities ---")
    print(f"  HUMAN: {probs.get('HUMAN', 0):.2%}")
    print(f"  AI_GENERATED: {probs.get('AI_GENERATED', 0):.2%}")
    
    # Description
    print(f"\n📝 Description: {result.get('description', 'N/A')}")
    
    # Details
    details = result.get("details", {})
    if details:
        print("\n--- Details ---")
        print(f"  Model: {details.get('model', 'N/A')}")
        print(f"  Duration: {details.get('audio_duration_seconds', 0):.2f}s")
        print(f"  Sample Rate: {details.get('sample_rate', 0)} Hz")
        print(f"  Device: {details.get('device', 'N/A')}")
    
    print("\n" + "=" * 60)


def demo_with_synthetic_audio():
    """Demo with synthetic audio (no file needed)"""
    import numpy as np
    from voiceguard_detector import VoiceGUARDDetector
    
    print("\n" + "=" * 60)
    print("DEMO: Synthetic Audio Classification")
    print("=" * 60)
    
    detector = VoiceGUARDDetector()
    detector.load_model()
    
    # Test 1: Pure sine wave (synthetic-like)
    print("\n📢 Test 1: Pure sine wave (440Hz)")
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    result1 = detector.classify(audio_sine, sample_rate)
    print_result(result1, "Pure Sine Wave")
    
    # Test 2: Complex audio with harmonics and variation
    print("\n📢 Test 2: Complex audio with natural variation")
    vibrato = 5 * np.sin(2 * np.pi * 5 * t)
    amplitude = 0.5 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
    audio_complex = amplitude * (
        np.sin(2 * np.pi * (200 + vibrato) * t) +
        0.5 * np.sin(2 * np.pi * 400 * t) +
        0.25 * np.sin(2 * np.pi * 600 * t)
    )
    audio_complex = (audio_complex + 0.02 * np.random.randn(len(t))).astype(np.float32)
    
    result2 = detector.classify(audio_complex, sample_rate)
    print_result(result2, "Complex Audio with Variation")
    
    # Test 3: Random noise
    print("\n📢 Test 3: Random noise")
    audio_noise = (0.3 * np.random.randn(int(sample_rate * duration))).astype(np.float32)
    
    result3 = detector.classify(audio_noise, sample_rate)
    print_result(result3, "Random Noise")
    
    print("\n✓ Demo complete!")


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("🛡️  VOICEGUARD - AI Voice Detection Demo")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Classify provided file(s)
        for file_path in sys.argv[1:]:
            if not os.path.exists(file_path):
                print(f"\n❌ File not found: {file_path}")
                continue
            
            print(f"\n📁 Processing: {file_path}")
            
            try:
                # Try API first
                result = classify_audio_file(file_path)
                print_result(result, Path(file_path).name)
            except requests.exceptions.ConnectionError:
                print("⚠️  API not running, using direct classification...")
                result = classify_with_detector(file_path)
                print_result(result, Path(file_path).name)
    else:
        # Run demo with synthetic audio
        print("\nNo audio file provided. Running demo with synthetic audio...")
        print("Usage: python demo_classify.py <audio_file.mp3>")
        demo_with_synthetic_audio()
    
    # Show sample files if available
    sample_dir = Path("test_samples")
    if sample_dir.exists():
        samples = list(sample_dir.glob("*.*"))
        if samples:
            print("\n📂 Available sample files:")
            for sample in samples:
                print(f"  - {sample}")
            print(f"\nRun: python demo_classify.py {samples[0]}")


if __name__ == "__main__":
    main()
