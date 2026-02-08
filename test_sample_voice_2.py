# test_sample_voice_2.py
"""Test the new sample voice_2.mp3 file"""

import requests
import base64

file_path = r"D:\development\workspace\HCL\HCL_GUVI\test_samples\sample voice_2.mp3"

print("\n" + "="*80)
print("TESTING: sample voice_2.mp3")
print("="*80)

# Load and encode
print("\nLoading audio file...")
with open(file_path, 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

import os
file_size = os.path.getsize(file_path)
print(f"✓ File size: {file_size:,} bytes")
print(f"✓ Base64 length: {len(audio_base64):,} characters")

print("\n⏳ Sending detection request to API...")

try:
    response = requests.post(
        'http://localhost:8000/api/voice-detection',
        json={
            'language': 'Tamil',
            'audioFormat': 'mp3',
            'audioBase64': audio_base64
        },
        timeout=60
    )
    
    result = response.json()
    
    print("\n" + "="*80)
    print("📊 DETECTION RESULTS")
    print("="*80)
    
    if result.get('status') == 'success':
        print(f"\n✅ Status: {result['status']}")
        print(f"\n🎯 Classification: {result['classification']}")
        print(f"📈 Confidence Score: {result['confidenceScore']:.2%}")
        print(f"⭐ Confidence Level: {result['confidenceLevel']}")
        print(f"🌐 Language: {result['language']}")
        
        print(f"\n💬 Explanation:")
        print(f"   {result['explanation']}")
        
        if 'details' in result:
            details = result['details']
            print(f"\n📊 Detailed Scores:")
            print(f"   Wav2Vec2 Score: {details.get('wav2vec2_score', 0):.4f}")
            print(f"   Acoustic Score: {details.get('acoustic_score', 0):.4f}")
            print(f"   Combined Score: {details.get('combined_score', 0):.4f}")
            
            print(f"\n🕐 Audio Info:")
            print(f"   Duration: {details.get('audio_duration_seconds', 0):.2f}s")
            print(f"   Sample Rate: {details.get('sample_rate', 0)}Hz")
            print(f"   Processing Time: {details.get('processing_time_ms', 0)}ms")
        
        # Final verdict
        print("\n" + "="*80)
        if result['classification'] == 'AI_GENERATED':
            print("⚠️  VERDICT: AI-GENERATED VOICE DETECTED")
        elif result['classification'] == 'HUMAN':
            print("✅ VERDICT: HUMAN VOICE DETECTED")
        else:
            print("❓ VERDICT: UNCERTAIN CLASSIFICATION")
        print("="*80)
        
    else:
        print(f"\n❌ Error: {result.get('message', 'Unknown error')}")
        if 'details' in result:
            print(f"Details: {result['details']}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n⚠️  Make sure the API server is running:")
    print("   python main.py")

print("\n")
