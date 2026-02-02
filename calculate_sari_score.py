import pickle
import sys
import requests
import json
import time
import re
from collections import Counter
import math
sys.stdout.reconfigure(encoding='utf-8')

def tokenize(text):
    """Simple tokenization for Sinhala text"""
    if not text:
        return []
    # Split by whitespace and remove empty strings
    tokens = [token.strip() for token in text.split() if token.strip()]
    return tokens

def calculate_bleu(prediction, reference, n=4):
    """Calculate BLEU score"""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Calculate precision for each n-gram
    precisions = []
    for i in range(1, n + 1):
        pred_ngrams = Counter()
        ref_ngrams = Counter()
        
        # Get n-grams from prediction
        for j in range(len(pred_tokens) - i + 1):
            ngram = tuple(pred_tokens[j:j + i])
            pred_ngrams[ngram] += 1
        
        # Get n-grams from reference
        for j in range(len(ref_tokens) - i + 1):
            ngram = tuple(ref_tokens[j:j + i])
            ref_ngrams[ngram] += 1
        
        # Calculate precision for this n-gram
        matches = 0
        for ngram in pred_ngrams:
            matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
        
        precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0
        precisions.append(precision)
    
    # Calculate brevity penalty
    if len(pred_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))
    else:
        bp = 1.0
    
    # Calculate BLEU
    if any(p == 0 for p in precisions):
        return 0.0
    
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return bleu

def calculate_sari(prediction, reference, source):
    """Calculate SARI score for text simplification"""
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))
    source_tokens = set(tokenize(source))
    
    # Addition: words in reference but not in source
    addition_ref = ref_tokens - source_tokens
    addition_pred = pred_tokens - source_tokens
    
    # Deletion: words in source but not in reference
    deletion_ref = source_tokens - ref_tokens
    deletion_pred = source_tokens - pred_tokens
    
    # Keep: words in both source and reference
    keep_ref = source_tokens & ref_tokens
    keep_pred = source_tokens & pred_tokens
    
    # Calculate F1 scores
    def f1_score(pred_set, ref_set):
        if not pred_set and not ref_set:
            return 1.0
        if not pred_set or not ref_set:
            return 0.0
        
        precision = len(pred_set & ref_set) / len(pred_set)
        recall = len(pred_set & ref_set) / len(ref_set)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    addition_f1 = f1_score(addition_pred, addition_ref)
    deletion_f1 = f1_score(deletion_pred, deletion_ref)
    keep_f1 = f1_score(keep_pred, keep_ref)
    
    # SARI is the average of the three F1 scores
    sari = (addition_f1 + deletion_f1 + keep_f1) / 3
    return sari * 100  # Convert to 0-100 scale

def test_improved_model():
    """Test the improved model and calculate new SARI score"""
    
    print("🧮 Calculating New SARI Score")
    print("=" * 50)
    
    # Load original results
    with open('sinhala_text_simplification_results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    sample_results = data.get('sample_results', [])
    
    # Test with all samples (or first 20 for faster testing)
    test_samples = sample_results[:20]  # Use first 20 samples
    
    print(f"Testing {len(test_samples)} samples...")
    
    improved_predictions = []
    original_predictions = []
    references = []
    sources = []
    
    for i, sample in enumerate(test_samples):
        print(f"Processing sample {i+1}/{len(test_samples)}...", end="\r")
        
        source_text = sample.get('input', '')
        reference_text = sample.get('reference', '')
        original_pred = sample.get('prediction', '')
        
        # Get improved prediction from API
        try:
            response = requests.post(
                'http://localhost:5000/api/simplify',
                json={'text': source_text},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                improved_pred = result.get('simplified', '')
                
                improved_predictions.append(improved_pred)
                original_predictions.append(original_pred)
                references.append(reference_text)
                sources.append(source_text)
            else:
                print(f"\nAPI Error for sample {i+1}: {response.status_code}")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"\nConnection error for sample {i+1}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(improved_predictions)} samples")
    
    if not improved_predictions:
        print("❌ No samples processed successfully")
        return
    
    # Calculate metrics
    print("\n📊 Calculating Metrics...")
    
    # Original metrics
    original_sari_scores = []
    original_bleu_scores = []
    
    # Improved metrics
    improved_sari_scores = []
    improved_bleu_scores = []
    
    for i in range(len(improved_predictions)):
        # Original scores
        orig_sari = calculate_sari(original_predictions[i], references[i], sources[i])
        orig_bleu = calculate_bleu(original_predictions[i], references[i])
        
        # Improved scores
        imp_sari = calculate_sari(improved_predictions[i], references[i], sources[i])
        imp_bleu = calculate_bleu(improved_predictions[i], references[i])
        
        original_sari_scores.append(orig_sari)
        original_bleu_scores.append(orig_bleu)
        improved_sari_scores.append(imp_sari)
        improved_bleu_scores.append(imp_bleu)
    
    # Calculate averages
    avg_original_sari = sum(original_sari_scores) / len(original_sari_scores)
    avg_original_bleu = sum(original_bleu_scores) / len(original_bleu_scores)
    avg_improved_sari = sum(improved_sari_scores) / len(improved_sari_scores)
    avg_improved_bleu = sum(improved_bleu_scores) / len(improved_bleu_scores)
    
    # Results
    print("\n🎯 SARI SCORE COMPARISON")
    print("=" * 40)
    print(f"Original SARI Score:  {avg_original_sari:.2f}/100")
    print(f"Improved SARI Score:  {avg_improved_sari:.2f}/100")
    print(f"SARI Improvement:     {avg_improved_sari - avg_original_sari:+.2f} points")
    print(f"SARI Improvement %:   {((avg_improved_sari - avg_original_sari) / avg_original_sari * 100):+.1f}%")
    
    print("\n📈 BLEU SCORE COMPARISON")
    print("=" * 40)
    print(f"Original BLEU Score:  {avg_original_bleu:.3f}")
    print(f"Improved BLEU Score:  {avg_improved_bleu:.3f}")
    print(f"BLEU Change:          {avg_improved_bleu - avg_original_bleu:+.3f}")
    
    print("\n📊 DETAILED ANALYSIS")
    print("=" * 40)
    
    # Length analysis
    orig_lengths = [len(tokenize(pred)) for pred in original_predictions]
    imp_lengths = [len(tokenize(pred)) for pred in improved_predictions]
    ref_lengths = [len(tokenize(ref)) for ref in references]
    
    avg_orig_length = sum(orig_lengths) / len(orig_lengths)
    avg_imp_length = sum(imp_lengths) / len(imp_lengths)
    avg_ref_length = sum(ref_lengths) / len(ref_lengths)
    
    print(f"Average Reference Length: {avg_ref_length:.1f} words")
    print(f"Average Original Length:  {avg_orig_length:.1f} words")
    print(f"Average Improved Length:  {avg_imp_length:.1f} words")
    print(f"Length Improvement:       {avg_imp_length - avg_orig_length:+.1f} words")
    print(f"Length Ratio (vs ref):    {avg_imp_length/avg_ref_length:.2f}")
    
    # Quality analysis
    orig_artifacts = sum(1 for pred in original_predictions if any(art in pred.lower() for art in ['<extra_id', 'sinhala', 'english']))
    imp_artifacts = sum(1 for pred in improved_predictions if any(art in pred.lower() for art in ['<extra_id', 'sinhala', 'english']))
    
    print(f"Original Artifacts:       {orig_artifacts}/{len(original_predictions)} samples")
    print(f"Improved Artifacts:       {imp_artifacts}/{len(improved_predictions)} samples")
    
    print("\n🎉 SUMMARY")
    print("=" * 40)
    if avg_improved_sari > avg_original_sari:
        print("✅ SARI Score IMPROVED!")
        print(f"   From {avg_original_sari:.2f} to {avg_improved_sari:.2f}")
        print(f"   Improvement: {avg_improved_sari - avg_original_sari:+.2f} points")
    else:
        print("⚠️  SARI Score needs more work")
    
    if avg_imp_length > avg_orig_length:
        print("✅ Output Length IMPROVED!")
        print(f"   From {avg_orig_length:.1f} to {avg_imp_length:.1f} words")
    
    if imp_artifacts < orig_artifacts:
        print("✅ Artifacts REDUCED!")
        print(f"   From {orig_artifacts} to {imp_artifacts} samples")

if __name__ == "__main__":
    print("Starting SARI score calculation...")
    print("Make sure the server is running: python run.py")
    print("Waiting 3 seconds...")
    time.sleep(3)
    test_improved_model()


