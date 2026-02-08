# convert_audio_to_base64.py

import base64
import sys
import os

def convert_audio_to_base64(audio_file_path):
    """Convert audio file to Base64 string"""
    
    if not os.path.exists(audio_file_path):
        print(f"❌ File not found: {audio_file_path}")
        return None
    
    try:
        # Read the audio file
        with open(audio_file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        # Convert to Base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print(f"✓ File: {audio_file_path}")
        print(f"✓ Size: {len(audio_bytes)} bytes")
        print(f"✓ Base64 length: {len(audio_base64)} characters")
        print(f"\n{'='*60}")
        print("Base64 String (copy this):")
        print(f"{'='*60}\n")
        print(audio_base64[:100] + "..." if len(audio_base64) > 100 else audio_base64)
        
        # Also save to file
        output_file = audio_file_path + ".base64.txt"
        with open(output_file, 'w') as f:
            f.write(audio_base64)
        
        print(f"\n✓ Full Base64 saved to: {output_file}")
        print(f"\nYou can now:")
        print(f"  1. Copy the Base64 from the file")
        print(f"  2. Paste it into http://localhost:8000/docs")
        print(f"  3. Or use it with requests/curl")
        
        return audio_base64
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AUDIO TO BASE64 CONVERTER")
    print("="*60 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage: python convert_audio_to_base64.py <audio_file>")
        print("\nExample:")
        print("  python convert_audio_to_base64.py my_voice.mp3")
        print("  python convert_audio_to_base64.py test.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    convert_audio_to_base64(audio_file)
