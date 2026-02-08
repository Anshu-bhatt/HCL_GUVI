"""
Test with synthetic AI-like audio (pure sine waves)
"""
import requests
import base64
import numpy as np
from scipy.io import wavfile
import tempfile
import os

API_URL = "http://localhost:8000/api/voice-detection"

print("\n" + "="*60)
print("🤖 TESTING: Synthetic AI Audio")
print("="*60 + "\n")

# Generate pure sine wave (very AI-like characteristics)
sample_rate = 16000
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.sin(2 * np.pi * 440 * t)  # Pure 440Hz tone
audio = (audio * 32767).astype(np.int16)

# Save to temp WAV file
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    temp_path = f.name
    wavfile.write(temp_path, sample_rate, audio)

# Convert to Base64
with open(temp_path, 'rb') as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

# Clean up temp file
os.unlink(temp_path)

print(f"✓ Generated synthetic audio: {duration}s pure sine wave")
print(f"✓ Base64 size: {len(audio_base64)} characters")

# Send to API
payload = {
    "language": "English",
    "audioFormat": "wav",
    "audioBase64": audio_base64
}

print(f"\n{'='*60}")
print("🚀 Sending to API...")
print(f"{'='*60}\n")

response = requests.post(API_URL, json=payload)
print(f"✓ Response: {response.status_code}\n")

if response.status_code == 200:
    result = response.json()
    
    print(f"{'='*60}")
    print("📊 RESULT")
    print(f"{'='*60}\n")
    
    classification = result.get('classification', 'N/A')
    confidence = result.get('confidenceScore', 0)
    confidence_level = result.get('confidenceLevel', 'N/A')
    combined_score = result.get('details', {}).get('combined_score', 0)
    
    # Emoji based on classification
    if classification == "AI_GENERATED":
        emoji = "🤖"
    elif classification == "HUMAN":
        emoji = "👤"
    else:
        emoji = "❓"
    
    print(f"{emoji} Classification: {classification}")
    print(f"📈 Confidence: {confidence:.2%}")
    print(f"🎯 Confidence Level: {confidence_level}")
    print(f"📊 Combined Score: {combined_score:.4f}")
    print(f"\n💬 Explanation:")
    print(f"   {result.get('explanation', 'N/A')}")
    
    print(f"\n📋 Details:")
    details = result.get('details', {})
    print(f"   • Wav2Vec2 Score: {details.get('wav2vec2_score', 0):.4f}")
    print(f"   • Acoustic Score: {details.get('acoustic_score', 0):.4f}")
    print(f"   • Processing Time: {details.get('processing_time_ms', 0)}ms")
    
    print(f"\n{'='*60}")
    print("✅ ANALYSIS")
    print(f"{'='*60}\n")
    
    if classification == "AI_GENERATED" and confidence_level == "HIGH":
        print("✅ PASS: Synthetic audio correctly detected as AI with HIGH confidence")
    elif classification == "AI_GENERATED" and confidence_level == "MEDIUM":
        print("🟡 ACCEPTABLE: Detected as AI with MEDIUM confidence")
    else:
        print("❌ UNEXPECTED: Synthetic audio should score high as AI")
        print(f"   Got: {classification} ({confidence_level})")
    
else:
    print(f"❌ API Error: {response.text}")

print()
