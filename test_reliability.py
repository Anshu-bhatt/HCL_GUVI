# test_reliability.py
"""Test the same audio multiple times to check consistency"""

import requests

API_URL = "http://localhost:8000"

print("\n" + "="*60)
print("🔍 RELIABILITY TEST - Multiple Runs")
print("="*60 + "\n")

# Read the Base64 file
base64_file = "test_samples/Namami_30s.mp3.base64.txt"

with open(base64_file, 'r') as f:
    audio_base64 = f.read()

print(f"Testing same audio 5 times to check consistency...\n")

results = []

for i in range(5):
    response = requests.post(
        f"{API_URL}/api/voice-detection",
        json={
            "language": "Hindi",
            "audioFormat": "mp3",
            "audioBase64": audio_base64
        },
        timeout=60
    )
    
    if response.status_code == 200:
        data = response.json()
        if data.get('status') != 'error':
            results.append({
                'run': i + 1,
                'classification': data['classification'],
                'confidence': data['confidenceScore'],
                'combined_score': data['details']['combined_score']
            })
            print(f"Run {i+1}: {data['classification']:12s} - {data['confidenceScore']:.2%} (score: {data['details']['combined_score']:.4f})")

print(f"\n{'='*60}")
print("📊 ANALYSIS")
print(f"{'='*60}\n")

if results:
    # Check consistency
    classifications = [r['classification'] for r in results]
    confidences = [r['confidence'] for r in results]
    scores = [r['combined_score'] for r in results]
    
    all_same = len(set(classifications)) == 1
    
    print(f"✓ Total runs: {len(results)}")
    print(f"✓ Classifications: {set(classifications)}")
    print(f"✓ Consistent result: {'YES ✓' if all_same else 'NO ✗'}")
    print(f"\n✓ Confidence range: {min(confidences):.2%} - {max(confidences):.2%}")
    print(f"✓ Average confidence: {sum(confidences)/len(confidences):.2%}")
    print(f"✓ Combined score range: {min(scores):.4f} - {max(scores):.4f}")
    
    # New confidence band analysis
    avg_score = sum(scores) / len(scores)
    
    print(f"\n{'='*60}")
    print("📈 CONFIDENCE BAND ANALYSIS")
    print(f"{'='*60}\n")
    
    print(f"✓ Average combined score: {avg_score:.4f}\n")
    print(f"📊 Confidence Bands (Adjusted for Practical Use):")
    print(f"   🟢 [0.60-1.00]: AI_GENERATED (HIGH confidence)")
    print(f"   🟡 [0.55-0.60): AI_GENERATED (MEDIUM confidence)")
    print(f"   ⚪ [0.45-0.55): UNCERTAIN (requires review)")
    print(f"   🟡 [0.30-0.45): HUMAN (MEDIUM confidence)")
    print(f"   🟢 [0.00-0.30): HUMAN (HIGH confidence)")
    
    if avg_score >= 0.60:
        print(f"\n✅ Status: Strong AI detection")
    elif avg_score >= 0.55:
        print(f"\n🟡 Status: Moderate AI detection")
    elif avg_score >= 0.45:
        print(f"\n⚪ Status: UNCERTAIN - borderline case")
        print(f"   → Manual review recommended")
    elif avg_score >= 0.30:
        print(f"\n🟡 Status: Moderate human detection ✓")
        print(f"   → Human voice detected")
    else:
        print(f"\n✅ Status: Strong human detection")
