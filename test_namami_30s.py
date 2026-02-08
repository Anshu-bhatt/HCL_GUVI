# test_namami_30s.py

import requests

API_URL = "http://localhost:8000"

print("\n" + "="*60)
print("🎤 TESTING: Namami Shamishan (30s)")
print("="*60 + "\n")

# Read the Base64 file
base64_file = "test_samples/Namami_30s.mp3.base64.txt"

print(f"Reading Base64 from: {base64_file}")
with open(base64_file, 'r') as f:
    audio_base64 = f.read()

print(f"✓ Base64 loaded: {len(audio_base64)} characters")
print(f"✓ Audio size: ~{len(audio_base64) * 3 // 4 / 1024:.1f} KB")

print(f"\n{'='*60}")
print(f"🚀 Sending to API (language: Hindi)...")
print(f"{'='*60}\n")

try:
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        json={
            "language": "Hindi",
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        },
        timeout=60
    )
    
    print(f"✓ Response: {response.status_code}\n")
    
    if response.status_code == 200:
        data = response.json()
        
        if data.get('status') == 'error':
            print(f"❌ API Error: {data.get('message')}")
        else:
            print(f"{'='*60}")
            print(f"📊 DETECTION RESULTS")
            print(f"{'='*60}\n")
            print(f"🎯 Classification: {data['classification']}")
            print(f"📈 Confidence: {data['confidenceScore']:.2%}")
            print(f"💬 Explanation: {data['explanation']}")
            
            if 'details' in data:
                print(f"\n📋 Technical Details:")
                print(f"  - Wav2Vec2 score: {data['details']['wav2vec2_score']:.4f}")
                print(f"  - Acoustic score: {data['details']['acoustic_score']:.4f}")
                print(f"  - Combined score: {data['details']['combined_score']:.4f}")
                print(f"  - Processing time: {data['details']['processing_time_ms']}ms")
                print(f"  - Audio duration: {data['details']['audio_duration_seconds']}s")
            
            print(f"\n{'='*60}\n")
            
            # Interpretation
            if data['classification'] == 'AI_GENERATED':
                print("🤖 This audio appears to be AI-generated")
            else:
                print("👤 This audio appears to be a human voice")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print(f"\n❌ Cannot connect to {API_URL}")
    print("Make sure the server is running: python main.py")
except Exception as e:
    print(f"\n❌ Error: {e}")
