# fix_long_audio.py
"""
Script to trim audio.wav to 30 seconds so it can be tested with the API
"""

import os
from pydub import AudioSegment

def trim_audio_file(input_file, output_file=None, max_duration_seconds=30):
    """
    Trim audio file to specified duration
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output file (optional, defaults to input_file with _trimmed suffix)
        max_duration_seconds: Maximum duration in seconds (default: 30)
    """
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return None
    
    try:
        # Detect file format
        file_ext = os.path.splitext(input_file)[1].lower()
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(input_file)
        elif file_ext == '.wav':
            audio = AudioSegment.from_wav(input_file)
        else:
            print(f"❌ Unsupported format: {file_ext}")
            return None
        
        # Get duration
        duration_seconds = len(audio) / 1000.0
        print(f"📊 Original duration: {duration_seconds:.2f}s")
        
        if duration_seconds <= max_duration_seconds:
            print(f"✅ Audio is already under {max_duration_seconds}s, no trimming needed")
            return input_file
        
        # Trim to max duration
        max_duration_ms = max_duration_seconds * 1000
        trimmed = audio[:max_duration_ms]
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_trimmed{file_ext}"
        
        # Export trimmed audio
        if file_ext == '.mp3':
            trimmed.export(output_file, format="mp3")
        elif file_ext == '.wav':
            trimmed.export(output_file, format="wav")
        
        new_duration = len(trimmed) / 1000.0
        file_size = os.path.getsize(output_file)
        
        print(f"✅ Trimmed audio saved!")
        print(f"   Output: {output_file}")
        print(f"   Duration: {new_duration:.2f}s")
        print(f"   Size: {file_size:,} bytes")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AUDIO TRIMMING TOOL")
    print("="*80 + "\n")
    
    # Trim the long audio file
    audio_file = r"D:\development\workspace\HCL\HCL_GUVI\test_samples\audio.wav"
    trimmed_file = trim_audio_file(audio_file, max_duration_seconds=30)
    
    if trimmed_file:
        print(f"\n✅ You can now test the trimmed file:")
        print(f"   {trimmed_file}")
