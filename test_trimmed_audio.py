# test_trimmed_audio.py
"""Quick test for the trimmed audio file"""

import requests
import base64

audio_file = r"D:\development\workspace\HCL\HCL_GUVI\test_samples\audio_trimmed.wav"

print("Loading audio file...")
with open(audio_file, 'rb') as f:
    audio_base64 = base64.b64encode(f.read()).decode('utf-8')

print(f"Audio size: {len(audio_base64)} characters")
print("Sending request to API...")

response = requests.post(
    'http://localhost:8000/api/voice-detection',
    json={
        'language': 'Tamil',
        'audioFormat': 'wav',
        'audioBase64': audio_base64
    },
    timeout=60
)

result = response.json()

print("\n" + "="*80)
print("RESULT FOR audio_trimmed.wav")
print("="*80)
print(f"Status: {result.get('status')}")
print(f"Classification: {result.get('classification')}")
print(f"Confidence: {result.get('confidenceScore', 0):.2%}")
print(f"Level: {result.get('confidenceLevel')}")
print(f"Explanation: {result.get('explanation')}")
print("="*80)
