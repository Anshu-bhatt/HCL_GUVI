from flask import Flask, render_template, request, jsonify
import base64
import os
import sys
import traceback
import librosa

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# We'll call the detection API directly instead of HTTP requests
from model_detector import HybridDetector
from audio_preprocessor import AudioProcessor
import soundfile as sf
import io
import numpy as np

# Initialize detectors
hybrid_detector = None
audio_processor = None

def init_models():
    global hybrid_detector, audio_processor
    if hybrid_detector is None:
        print("🔄 Loading AI detection models...")
        hybrid_detector = HybridDetector()
        audio_processor = AudioProcessor()
        print("✅ Models loaded successfully!")

init_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read audio file
        audio_data = audio_file.read()
        
        # Process audio
        print(f"📁 Processing file: {audio_file.filename}")
        audio, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Handle stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Check duration
        duration = len(audio) / sample_rate
        print(f"⏱️ Audio duration: {duration:.2f}s")
        
        if duration > 30:
            return jsonify({
                'error': f'Audio too long ({duration:.2f}s). Maximum 30 seconds allowed.'
            }), 400
        
        # Resample to target sample rate if needed
        if sample_rate != audio_processor.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=audio_processor.sample_rate)
            sample_rate = audio_processor.sample_rate
        
        # Preprocess audio
        audio_processed = audio_processor.preprocess_audio(audio)
        
        # Extract acoustic features
        acoustic_features = audio_processor.extract_features(audio)
        
        # Detect
        import time
        start_time = time.time()
        classification, confidence, details = hybrid_detector.detect(
            audio_processed, 
            acoustic_features, 
            sample_rate
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Determine confidence level
        if confidence >= 0.70:
            confidence_level = "HIGH"
        elif confidence >= 0.55:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        # Build response
        result = {
            'status': 'success',
            'classification': classification,
            'confidence': float(confidence),
            'confidence_level': confidence_level,
            'details': {
                'wav2vec2_score': float(details.get('wav2vec2_score', 0)),
                'acoustic_score': float(details.get('acoustic_score', 0)),
                'combined_score': float(details.get('combined_score', 0)),
                'processing_time_ms': float(processing_time)
            },
            'explanation': get_explanation(classification, confidence)
        }
        
        print(f"✅ Detection complete: {classification} ({confidence*100:.1f}%)")
        return jsonify(result)
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        traceback.print_exc()
        return jsonify({'error': error_msg, 'trace': traceback.format_exc()}), 500

def get_explanation(classification, confidence):
    if classification == 'HUMAN':
        if confidence > 0.7:
            return "Detected as human voice with high confidence due to natural acoustic patterns and irregular variations."
        else:
            return "Detected as human voice with moderate confidence due to irregular acoustic patterns."
    elif classification == 'AI_GENERATED':
        if confidence > 0.7:
            return "Detected as AI-generated voice with high confidence due to consistent patterns typical of synthetic speech."
        else:
            return "Detected as AI-generated voice with moderate confidence due to some synthetic indicators."
    else:
        return "Uncertain classification. The audio shows characteristics of both human and AI-generated speech."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
