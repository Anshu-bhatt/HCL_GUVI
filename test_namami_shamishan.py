# test_namami_shamishan.py

import requests
import base64

API_URL = "http://localhost:8000"

def test_namami_audio():
    """Test the Namami Shamishan audio file"""
    
    print("\n" + "="*60)
    print("🎤 TESTING: Namami Shamishan.mp3")
    print("="*60 + "\n")
    
    # Read the Base64 file
    base64_file = "test_samples/Namami Shamishan.mp3.base64.txt"
    
    print(f"Reading Base64 from: {base64_file}")
    with open(base64_file, 'r') as f:
        audio_base64 = f.read()
    
    print(f"✓ Base64 loaded: {len(audio_base64)} characters")
    print(f"✓ Original audio size: ~{len(audio_base64) * 3 // 4 / 1024:.1f} KB")
    
    # Test with different languages
    languages = ["Hindi", "English", "Tamil"]
    
    for language in languages:
        print(f"\n{'='*60}")
        print(f"Testing with language: {language}")
        print(f"{'='*60}\n")
        
        print(f"🚀 Sending request...")
        
        try:
            response = requests.post(
                f"{API_URL}/api/voice-detection",
                json={
                    "language": language,
                    "audioFormat": "mp3",
                    "audioBase64": audio_base64
                },
                timeout=120  # Longer timeout for large file
            )
            
            print(f"✓ Response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if it's an error response
                if data.get('status') == 'error':
                    print(f"\n❌ API Error:")
                    print(f"  Message: {data.get('message', 'Unknown error')}")
                    if 'details' in data:
                        print(f"  Details: {data['details']}")
                    continue
                
                print(f"\n📊 RESULTS:")
                print(f"  🎯 Classification: {data['classification']}")
                print(f"  📈 Confidence: {data['confidenceScore']:.2%}")
                print(f"  💬 {data['explanation']}")
                
                if 'details' in data:
                    print(f"\n  📋 Details:")
                    print(f"     - Wav2Vec2 score: {data['details']['wav2vec2_score']:.4f}")
                    print(f"     - Acoustic score: {data['details']['acoustic_score']:.4f}")
                    print(f"     - Combined score: {data['details']['combined_score']:.4f}")
                    print(f"     - Processing time: {data['details']['processing_time_ms']}ms")
                    print(f"     - Audio duration: {data['details']['audio_duration_seconds']}s")
                
            else:
                print(f"\n❌ Error: {response.status_code}")
                error_data = response.json()
                if 'message' in error_data:
                    print(f"Message: {error_data['message']}")
                else:
                    print(response.text)
                
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to {API_URL}")
            print("Make sure the server is running: python main.py")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_namami_audio()
