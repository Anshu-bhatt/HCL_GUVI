# download_sample.py

import urllib.request
import os

def download_sample_audio():
    """
    Download a sample MP3 for testing
    Using a public domain audio file
    """
    
    # Sample audio URL (public domain)
    url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
    output_path = "test_samples/sample_voice.mp3"
    
    print("Downloading sample audio...")
    print(f"URL: {url}")
    print(f"Saving to: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded successfully!")
        print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nAlternative: Record your own voice:")
        print("1. Record a 5-10 second voice message on your phone")
        print("2. Save as MP3")
        print("3. Place in test_samples/sample_voice.mp3")

if __name__ == "__main__":
    download_sample_audio()