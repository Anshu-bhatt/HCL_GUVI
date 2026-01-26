# utils/audio_helpers.py

import base64
import os

def file_to_base64(filepath: str) -> str:
    """
    Convert audio file to Base64 string
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Base64 encoded string
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "rb") as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    return audio_base64

def base64_to_file(audio_base64: str, output_path: str) -> None:
    """
    Save Base64 audio to file
    
    Args:
        audio_base64: Base64 encoded audio
        output_path: Where to save the file
    """
    audio_bytes = base64.b64decode(audio_base64)
    
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    
    print(f"Saved to: {output_path}")

def get_audio_duration(filepath: str) -> float:
    """
    Get duration of audio file in seconds
    
    Args:
        filepath: Path to audio file
        
    Returns:
        Duration in seconds
    """
    import librosa
    audio, sr = librosa.load(filepath, sr=None)
    duration = len(audio) / sr
    return duration

# Example usage functions
if __name__ == "__main__":
    # Example: Convert MP3 to Base64
    # audio_base64 = file_to_base64("sample.mp3")
    # print(f"Base64 length: {len(audio_base64)}")
    
    print("Audio helper utilities loaded")
    print("Available functions:")
    print("  - file_to_base64(filepath)")
    print("  - base64_to_file(audio_base64, output_path)")
    print("  - get_audio_duration(filepath)")