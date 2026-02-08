# trim_audio.py

import librosa
import soundfile as sf
import sys

def trim_audio(input_file, output_file, max_duration=30):
    """Trim audio to max duration"""
    
    print(f"\n{'='*60}")
    print(f"TRIMMING AUDIO")
    print(f"{'='*60}\n")
    
    print(f"Loading: {input_file}")
    audio, sr = librosa.load(input_file, sr=None, mono=True)
    
    duration = len(audio) / sr
    print(f"✓ Original duration: {duration:.1f}s")
    print(f"✓ Sample rate: {sr}Hz")
    
    if duration <= max_duration:
        print(f"✓ Audio is already under {max_duration}s")
        return
    
    # Trim to max_duration
    max_samples = int(max_duration * sr)
    audio_trimmed = audio[:max_samples]
    
    print(f"✓ Trimmed to: {max_duration}s")
    print(f"✓ Saving to: {output_file}")
    
    sf.write(output_file, audio_trimmed, sr)
    
    print(f"\n✅ Done! You can now test with:")
    print(f"   python convert_audio_to_base64.py \"{output_file}\"")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trim_audio.py <input_file> [output_file] [duration]")
        print("\nExample:")
        print('  python trim_audio.py "test_samples/Namami Shamishan.mp3" "test_samples/Namami_30s.mp3" 30')
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.mp3', '_30s.mp3')
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    
    trim_audio(input_file, output_file, duration)
