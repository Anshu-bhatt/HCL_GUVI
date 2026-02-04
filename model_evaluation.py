#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for VoiceGUARD Bias-Corrected Model
Calculates accuracy, precision, recall, F1-score, and confusion matrix
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append('.')

try:
    import librosa
    from voiceguard_detector import VoiceGUARDDetector
    HAS_MODEL = True
except ImportError as e:
    print(f"⚠️  Model import failed: {e}")
    HAS_MODEL = False

class ModelEvaluator:
    def __init__(self):
        self.detector = None
        self.evaluation_results = {}
        self.detailed_results = []
        
    def initialize_model(self):
        """Initialize the VoiceGUARD detector"""
        if not HAS_MODEL:
            raise ImportError("VoiceGUARD model dependencies not available")
            
        print("🔧 Initializing VoiceGUARD Detector...")
        self.detector = VoiceGUARDDetector()
        
        try:
            self.detector.load_model()
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def create_test_dataset(self):
        """Create or locate test dataset with labeled samples"""
        print("📂 Preparing test dataset...")
        
        test_cases = []
        
        # Real human voice samples (you would replace these with actual human recordings)
        human_samples = [
            {
                "type": "human",
                "description": "Original human recording",
                "confidence_range": (0.6, 0.95),
                "expected": "HUMAN"
            }
        ]
        
        # AI-generated samples (you would replace these with actual AI samples)
        ai_samples = [
            {
                "type": "ai", 
                "description": "ElevenLabs generated voice",
                "confidence_range": (0.6, 0.95),
                "expected": "AI_GENERATED"
            }
        ]
        
        # Synthetic test cases for comprehensive evaluation
        print("🎯 Creating synthetic test scenarios...")
        
        # Test case 1: Clear human speech characteristics
        test_cases.append({
            "name": "Human Speech Pattern",
            "expected": "HUMAN",
            "features": {
                "pitch_variation": "high",
                "formant_clarity": "high", 
                "natural_pauses": "present"
            }
        })
        
        # Test case 2: AI-like characteristics
        test_cases.append({
            "name": "AI Speech Pattern",
            "expected": "AI_GENERATED",
            "features": {
                "pitch_variation": "low",
                "formant_clarity": "artificial",
                "natural_pauses": "absent"
            }
        })
        
        return test_cases
    
    def evaluate_existing_files(self):
        """Evaluate any existing audio files in the workspace"""
        results = []
        
        # Look for audio files
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.m4a', '*.flac']:
            audio_files.extend(Path('.').rglob(ext))
        
        if not audio_files:
            print("⚠️  No audio files found for evaluation")
            return results
            
        print(f"🎵 Found {len(audio_files)} audio files for evaluation")
        
        for file_path in audio_files:
            try:
                print(f"   🔍 Processing: {file_path.name}")
                
                # Load audio
                audio, sr = librosa.load(str(file_path), sr=16000, mono=True)
                
                # Get prediction
                start_time = time.time()
                result = self.detector.classify(audio, sr)
                inference_time = time.time() - start_time
                
                # Extract metrics
                classification = result['classification']
                confidence = result['confidence']
                details = result['details']
                
                file_result = {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'classification': classification,
                    'confidence': confidence,
                    'inference_time_ms': round(inference_time * 1000, 2),
                    'audio_duration': round(len(audio) / sr, 2),
                    'ast_score': details.get('ast_prob_real', 0),
                    'acoustic_score': details.get('acoustic_score', 0),
                    'ensemble_method': details.get('ensemble_method', 'Unknown')
                }
                
                results.append(file_result)
                
                # Display result
                emoji = "🟢" if classification == "HUMAN" else "🔴"
                print(f"      {emoji} {classification} ({confidence:.1%}) in {inference_time:.2f}s")
                
            except Exception as e:
                print(f"      ❌ Error processing {file_path.name}: {e}")
                
        return results
    
    def calculate_performance_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive performance metrics"""
        
        # Convert string labels to binary
        y_true_binary = [1 if label == "HUMAN" else 0 for label in y_true]
        y_pred_binary = [1 if label == "HUMAN" else 0 for label in y_pred]
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, average='binary')
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, average='binary')
        metrics['f1_score'] = f1_score(y_true_binary, y_pred_binary, average='binary')
        
        # Class-specific metrics
        metrics['precision_human'] = precision_score(y_true_binary, y_pred_binary, pos_label=1)
        metrics['recall_human'] = recall_score(y_true_binary, y_pred_binary, pos_label=1)
        metrics['precision_ai'] = precision_score(y_true_binary, y_pred_binary, pos_label=0)
        metrics['recall_ai'] = recall_score(y_true_binary, y_pred_binary, pos_label=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC AUC if probabilities available
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true_binary, y_prob)
            except:
                metrics['roc_auc'] = None
        
        # Additional bias metrics
        human_count = sum(y_true_binary)
        ai_count = len(y_true_binary) - human_count
        
        if human_count > 0 and ai_count > 0:
            # Check for bias
            human_accuracy = sum([1 for i, (true, pred) in enumerate(zip(y_true_binary, y_pred_binary)) 
                                 if true == 1 and true == pred]) / human_count
            ai_accuracy = sum([1 for i, (true, pred) in enumerate(zip(y_true_binary, y_pred_binary)) 
                              if true == 0 and true == pred]) / ai_count
            
            metrics['human_accuracy'] = human_accuracy
            metrics['ai_accuracy'] = ai_accuracy
            metrics['bias_score'] = abs(human_accuracy - ai_accuracy)  # Lower is better
        
        return metrics
    
    def run_synthetic_evaluation(self):
        """Run evaluation on synthetic test cases"""
        print("🧪 Running Synthetic Evaluation Tests...")
        
        synthetic_results = []
        
        # Test 1: Bias Check - Multiple identical samples
        print("   📊 Test 1: Bias Consistency Check")
        for i in range(5):
            # Generate simple sine wave (should be neutral)
            duration = 2.0
            sr = 16000
            t = np.linspace(0, duration, int(duration * sr))
            frequency = 440 + (i * 50)  # Vary frequency slightly
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            result = self.detector.classify(audio, sr)
            synthetic_results.append({
                'test_type': 'bias_check',
                'sample_id': i,
                'classification': result['classification'],
                'confidence': result['confidence'],
                'frequency': frequency
            })
        
        # Test 2: Edge Cases
        print("   🔬 Test 2: Edge Case Analysis")
        
        # Very short audio
        short_audio = np.random.normal(0, 0.1, int(0.6 * 16000))  # 0.6 seconds
        try:
            result = self.detector.classify(short_audio, 16000)
            synthetic_results.append({
                'test_type': 'edge_case',
                'case': 'short_audio',
                'classification': result['classification'],
                'confidence': result['confidence']
            })
        except Exception as e:
            synthetic_results.append({
                'test_type': 'edge_case', 
                'case': 'short_audio',
                'error': str(e)
            })
        
        # Silent audio
        silent_audio = np.zeros(int(2.0 * 16000))
        try:
            result = self.detector.classify(silent_audio, 16000)
            synthetic_results.append({
                'test_type': 'edge_case',
                'case': 'silent_audio',
                'classification': result['classification'],
                'confidence': result['confidence']
            })
        except Exception as e:
            synthetic_results.append({
                'test_type': 'edge_case',
                'case': 'silent_audio', 
                'error': str(e)
            })
        
        # White noise
        noise_audio = np.random.normal(0, 0.1, int(2.0 * 16000))
        try:
            result = self.detector.classify(noise_audio, 16000)
            synthetic_results.append({
                'test_type': 'edge_case',
                'case': 'white_noise',
                'classification': result['classification'], 
                'confidence': result['confidence']
            })
        except Exception as e:
            synthetic_results.append({
                'test_type': 'edge_case',
                'case': 'white_noise',
                'error': str(e)
            })
        
        return synthetic_results
    
    def generate_evaluation_report(self, file_results, synthetic_results):
        """Generate comprehensive evaluation report"""
        
        report = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_name': 'VoiceGUARD Bias-Corrected',
                'total_files_evaluated': len(file_results),
                'synthetic_tests_run': len(synthetic_results)
            },
            'file_evaluation': {
                'results': file_results,
                'summary': self._summarize_file_results(file_results)
            },
            'synthetic_evaluation': {
                'results': synthetic_results,
                'summary': self._summarize_synthetic_results(synthetic_results)
            },
            'bias_analysis': self._analyze_bias(file_results, synthetic_results),
            'performance_analysis': self._analyze_performance(file_results)
        }
        
        return report
    
    def _summarize_file_results(self, results):
        """Summarize file evaluation results"""
        if not results:
            return {"message": "No files evaluated"}
            
        human_count = sum(1 for r in results if r['classification'] == 'HUMAN')
        ai_count = sum(1 for r in results if r['classification'] == 'AI_GENERATED')
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_inference_time = np.mean([r['inference_time_ms'] for r in results])
        
        return {
            'human_classifications': human_count,
            'ai_classifications': ai_count,
            'avg_confidence': round(avg_confidence, 3),
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'confidence_distribution': {
                'min': round(min([r['confidence'] for r in results]), 3),
                'max': round(max([r['confidence'] for r in results]), 3),
                'std': round(np.std([r['confidence'] for r in results]), 3)
            }
        }
    
    def _summarize_synthetic_results(self, results):
        """Summarize synthetic test results"""
        bias_tests = [r for r in results if r.get('test_type') == 'bias_check']
        edge_tests = [r for r in results if r.get('test_type') == 'edge_case']
        
        summary = {}
        
        if bias_tests:
            classifications = [r['classification'] for r in bias_tests if 'classification' in r]
            if classifications:
                human_pct = (classifications.count('HUMAN') / len(classifications)) * 100
                summary['bias_check'] = {
                    'total_tests': len(bias_tests),
                    'human_percentage': round(human_pct, 1),
                    'ai_percentage': round(100 - human_pct, 1),
                    'bias_detected': abs(human_pct - 50) > 20  # >20% deviation suggests bias
                }
        
        if edge_tests:
            successful_tests = [r for r in edge_tests if 'classification' in r]
            failed_tests = [r for r in edge_tests if 'error' in r]
            summary['edge_cases'] = {
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'results': {r['case']: r.get('classification', 'ERROR') for r in edge_tests}
            }
        
        return summary
    
    def _analyze_bias(self, file_results, synthetic_results):
        """Analyze potential bias in the model"""
        analysis = {}
        
        # File-based bias analysis
        if file_results:
            classifications = [r['classification'] for r in file_results]
            human_pct = (classifications.count('HUMAN') / len(classifications)) * 100
            
            analysis['file_bias'] = {
                'human_percentage': round(human_pct, 1),
                'ai_percentage': round(100 - human_pct, 1),
                'potential_bias': 'AI_BIASED' if human_pct < 30 else 'HUMAN_BIASED' if human_pct > 70 else 'BALANCED'
            }
        
        # Synthetic bias analysis
        synthetic_bias_tests = [r for r in synthetic_results if r.get('test_type') == 'bias_check']
        if synthetic_bias_tests:
            synth_classifications = [r['classification'] for r in synthetic_bias_tests if 'classification' in r]
            if synth_classifications:
                synth_human_pct = (synth_classifications.count('HUMAN') / len(synth_classifications)) * 100
                analysis['synthetic_bias'] = {
                    'human_percentage': round(synth_human_pct, 1),
                    'bias_score': round(abs(synth_human_pct - 50), 1),  # Distance from 50%
                    'bias_status': 'SIGNIFICANT' if abs(synth_human_pct - 50) > 20 else 'MINIMAL'
                }
        
        return analysis
    
    def _analyze_performance(self, file_results):
        """Analyze performance characteristics"""
        if not file_results:
            return {"message": "No performance data available"}
            
        inference_times = [r['inference_time_ms'] for r in file_results]
        confidences = [r['confidence'] for r in file_results]
        
        return {
            'inference_performance': {
                'avg_time_ms': round(np.mean(inference_times), 2),
                'min_time_ms': round(min(inference_times), 2),
                'max_time_ms': round(max(inference_times), 2),
                'std_time_ms': round(np.std(inference_times), 2)
            },
            'confidence_analysis': {
                'avg_confidence': round(np.mean(confidences), 3),
                'confidence_std': round(np.std(confidences), 3),
                'low_confidence_count': sum(1 for c in confidences if c < 0.6),
                'high_confidence_count': sum(1 for c in confidences if c > 0.8)
            }
        }
    
    def display_console_report(self, report):
        """Display formatted evaluation report in console"""
        
        print("\n" + "="*80)
        print("🎯 VOICEGUARD MODEL EVALUATION REPORT")
        print("="*80)
        
        # Metadata
        metadata = report['evaluation_metadata']
        print(f"📅 Evaluation Time: {metadata['timestamp']}")
        print(f"🤖 Model: {metadata['model_name']}")
        print(f"📊 Files Evaluated: {metadata['total_files_evaluated']}")
        print(f"🧪 Synthetic Tests: {metadata['synthetic_tests_run']}")
        
        # File Evaluation Summary
        print(f"\n{'📁 FILE EVALUATION SUMMARY':<40}")
        print("-" * 50)
        
        file_summary = report['file_evaluation']['summary']
        if 'human_classifications' in file_summary:
            print(f"🟢 Human Classifications: {file_summary['human_classifications']}")
            print(f"🔴 AI Classifications: {file_summary['ai_classifications']}")
            print(f"📈 Average Confidence: {file_summary['avg_confidence']:.1%}")
            print(f"⚡ Average Inference Time: {file_summary['avg_inference_time_ms']:.1f}ms")
            
            conf_dist = file_summary['confidence_distribution']
            print(f"📊 Confidence Range: {conf_dist['min']:.1%} - {conf_dist['max']:.1%} (σ={conf_dist['std']:.3f})")
        else:
            print("⚠️  No file evaluation data available")
        
        # Bias Analysis
        print(f"\n{'🎯 BIAS ANALYSIS':<40}")
        print("-" * 50)
        
        bias_analysis = report['bias_analysis']
        if 'file_bias' in bias_analysis:
            file_bias = bias_analysis['file_bias']
            print(f"📂 File-based Bias:")
            print(f"   🟢 Human: {file_bias['human_percentage']}%")
            print(f"   🔴 AI: {file_bias['ai_percentage']}%") 
            print(f"   📊 Status: {file_bias['potential_bias']}")
        
        if 'synthetic_bias' in bias_analysis:
            synth_bias = bias_analysis['synthetic_bias']
            print(f"🧪 Synthetic Bias Test:")
            print(f"   📈 Bias Score: {synth_bias['bias_score']}% (deviation from 50%)")
            print(f"   📊 Status: {synth_bias['bias_status']}")
        
        # Performance Analysis
        print(f"\n{'⚡ PERFORMANCE ANALYSIS':<40}")
        print("-" * 50)
        
        perf_analysis = report['performance_analysis']
        if 'inference_performance' in perf_analysis:
            inf_perf = perf_analysis['inference_performance']
            conf_analysis = perf_analysis['confidence_analysis']
            
            print(f"🚀 Inference Speed:")
            print(f"   📊 Average: {inf_perf['avg_time_ms']:.1f}ms")
            print(f"   📈 Range: {inf_perf['min_time_ms']:.1f}ms - {inf_perf['max_time_ms']:.1f}ms")
            
            print(f"🎯 Confidence Distribution:")
            print(f"   📊 Average: {conf_analysis['avg_confidence']:.1%}")
            print(f"   📉 Low Confidence (<60%): {conf_analysis['low_confidence_count']} samples")
            print(f"   📈 High Confidence (>80%): {conf_analysis['high_confidence_count']} samples")
        
        # Synthetic Test Results
        print(f"\n{'🧪 SYNTHETIC TEST RESULTS':<40}")
        print("-" * 50)
        
        synth_summary = report['synthetic_evaluation']['summary']
        if 'bias_check' in synth_summary:
            bias_check = synth_summary['bias_check']
            print(f"🎯 Bias Consistency Test:")
            print(f"   📊 Tests Run: {bias_check['total_tests']}")
            print(f"   🟢 Human: {bias_check['human_percentage']}%")
            print(f"   🔴 AI: {bias_check['ai_percentage']}%")
            bias_status = "✅ PASS" if not bias_check['bias_detected'] else "❌ BIAS DETECTED"
            print(f"   📈 Result: {bias_status}")
        
        if 'edge_cases' in synth_summary:
            edge_cases = synth_summary['edge_cases']
            print(f"🔬 Edge Case Tests:")
            print(f"   ✅ Successful: {edge_cases['successful_tests']}")
            print(f"   ❌ Failed: {edge_cases['failed_tests']}")
            for case, result in edge_cases['results'].items():
                print(f"      • {case}: {result}")
        
        # Overall Assessment
        print(f"\n{'🏆 OVERALL ASSESSMENT':<40}")
        print("-" * 50)
        
        # Determine overall grade
        issues = []
        
        # Check for bias issues
        if 'file_bias' in bias_analysis:
            if bias_analysis['file_bias']['potential_bias'] != 'BALANCED':
                issues.append(f"File classification bias: {bias_analysis['file_bias']['potential_bias']}")
        
        if 'synthetic_bias' in bias_analysis:
            if bias_analysis['synthetic_bias']['bias_status'] == 'SIGNIFICANT':
                issues.append(f"Synthetic test bias detected")
        
        # Check performance issues  
        if 'confidence_analysis' in perf_analysis:
            low_conf_pct = (perf_analysis['confidence_analysis']['low_confidence_count'] / 
                          metadata['total_files_evaluated'] * 100) if metadata['total_files_evaluated'] > 0 else 0
            if low_conf_pct > 50:
                issues.append("High percentage of low-confidence predictions")
        
        if not issues:
            grade = "🏆 EXCELLENT"
            print(f"{grade} - Model shows balanced, unbiased classification")
        elif len(issues) == 1:
            grade = "✅ GOOD"
            print(f"{grade} - Minor issues detected:")
            for issue in issues:
                print(f"   ⚠️  {issue}")
        else:
            grade = "⚠️  NEEDS IMPROVEMENT"
            print(f"{grade} - Multiple issues detected:")
            for issue in issues:
                print(f"   ❌ {issue}")
        
        print("\n" + "="*80)
        
        return grade
    
    def save_detailed_report(self, report):
        """Save detailed report to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voiceguard_evaluation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"📄 Detailed report saved to: {filename}")
        return filename
    
    def run_complete_evaluation(self):
        """Run the complete evaluation process"""
        
        print("🎯 Starting VoiceGUARD Model Evaluation")
        print("="*50)
        
        # Initialize model
        if not self.initialize_model():
            print("❌ Cannot proceed without model")
            return None
        
        # Evaluate existing files
        print("\n📁 Evaluating existing audio files...")
        file_results = self.evaluate_existing_files()
        
        # Run synthetic tests
        print("\n🧪 Running synthetic evaluation tests...")
        synthetic_results = self.run_synthetic_evaluation()
        
        # Generate comprehensive report
        print("\n📊 Generating evaluation report...")
        report = self.generate_evaluation_report(file_results, synthetic_results)
        
        # Display console report
        grade = self.display_console_report(report)
        
        # Save detailed report
        report_file = self.save_detailed_report(report)
        
        return {
            'grade': grade,
            'report': report,
            'report_file': report_file
        }

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    
    try:
        results = evaluator.run_complete_evaluation()
        
        if results:
            print(f"\n🎉 Evaluation completed successfully!")
            print(f"📊 Overall Grade: {results['grade']}")
            print(f"📄 Report File: {results['report_file']}")
        else:
            print(f"\n❌ Evaluation failed")
            
    except Exception as e:
        print(f"\n💥 Evaluation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()