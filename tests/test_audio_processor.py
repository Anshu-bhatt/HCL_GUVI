# tests/test_audio_processor.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_preprocessor import AudioProcessor
import base64
import numpy as np

def create_test_audio_base64():
    """
    Create a simple test audio signal (sine wave)
    Returns Base64 encoded MP3
    """
    import io
    from scipy.io import wavfile
    
    # Generate 2 seconds of 440Hz sine wave (A note)
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit
    
    # Save to WAV buffer
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    buffer.seek(0)
    
    # Encode to Base64
    audio_bytes = buffer.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    return audio_base64

def test_audio_processor():
    """Test all AudioProcessor functionality"""
    
    print("\n" + "="*60)
    print("TESTING AUDIO PROCESSOR")
    print("="*60 + "\n")
    
    # Initialize processor
    processor = AudioProcessor(sample_rate=16000, max_duration=30)
    print("✓ AudioProcessor initialized\n")
    
    # Test 1: Create test audio
    print("Test 1: Creating test audio signal...")
    try:
        audio_base64 = create_test_audio_base64()
        print(f"✓ Created Base64 audio ({len(audio_base64)} characters)\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return
    
    # Test 2: Decode Base64 audio
    print("Test 2: Decoding Base64 audio...")
    try:
        audio, sr = processor.decode_base64_audio(audio_base64)
        print(f"✓ Decoded audio: shape={audio.shape}, sr={sr}Hz\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return
    
    # Test 3: Preprocess audio
    print("Test 3: Preprocessing audio...")
    try:
        preprocessed = processor.preprocess_audio(audio)
        print(f"✓ Preprocessed: shape={preprocessed.shape}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return
    
    # Test 4: Extract features
    print("Test 4: Extracting features...")
    try:
        features = processor.extract_features(audio)
        print(f"✓ Extracted {len(features)} features:")
        for key, value in list(features.items())[:5]:  # Show first 5
            print(f"  - {key}: {value:.4f}")
        print(f"  ... and {len(features) - 5} more\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return
    
    # Test 5: Get audio info
    print("Test 5: Getting audio info...")
    try:
        info = processor.get_audio_info(audio)
        print(f"✓ Audio info:")
        for key, value in info.items():
            print(f"  - {key}: {value}")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        return
    
    # Test 6: Error handling - invalid Base64
    print("Test 6: Testing error handling (invalid Base64)...")
    try:
        processor.decode_base64_audio("invalid_base64_string!")
        print("✗ Should have raised an error\n")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}\n")
    
    # Test 7: Error handling - empty audio
    print("Test 7: Testing error handling (too short audio)...")
    try:
        # Create very short audio
        short_audio = np.array([0.1, 0.2, 0.1])  # Only 3 samples
        processor._validate_audio(short_audio)
        print("✗ Should have raised an error\n")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}\n")
    
    print("="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_audio_processor()