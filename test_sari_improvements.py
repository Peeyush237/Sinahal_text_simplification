"""
Comprehensive SARI Improvement Testing Script
Tests the improved SARI-optimized generation system
"""
import pickle
import sys
import requests
import json
import time
from calculate_sari_score import calculate_sari, calculate_bleu, tokenize
sys.stdout.reconfigure(encoding='utf-8')


def check_server_running(max_retries=10, retry_delay=3):
    """
    Check if Flask server is running, wait for it if not
    
    Returns:
        bool: True if server is running, False otherwise
    """
    print("\n🔍 Checking if server is running...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get('http://localhost:5000/api/health', timeout=2)
            if response.status_code == 200:
                data = response.json()
                print("✅ Server is running!")
                print(f"   Status: {data.get('status')}")
                print(f"   Model loaded: {data.get('model_loaded')}")
                print(f"   Data loaded: {data.get('data_loaded')}")
                return True
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"   ⏳ Server not ready yet... waiting {retry_delay} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print("❌ Server is NOT running after waiting")
                return False
        except Exception as e:
            print(f"⚠️  Error checking server: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False
    
    return False


def test_sari_improvements():
    """Test the improved SARI optimization system"""
    
    print("🚀 Testing Improved SARI Optimization System")
    print("=" * 60)
    
    # Check if server is running
    if not check_server_running():
        print("\n" + "=" * 60)
        print("❌ ERROR: Server is not running!")
        print("=" * 60)
        print("\n📋 To fix this:")
        print("1. Open a NEW terminal/command prompt")
        print("2. Navigate to the project directory:")
        print("   cd sinhala-text-simplifier\\sinhala-text-simplifier")
        print("3. Start the server:")
        print("   python start_server.py")
        print("   OR")
        print("   python run.py")
        print("4. Wait until you see: 'Running on http://0.0.0.0:5000'")
        print("5. Keep that terminal open and run this test script again")
        print("\n" + "=" * 60)
        return
    
    # Load original results
    try:
        with open('sinhala_text_simplification_results.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: sinhala_text_simplification_results.pkl not found")
        return
    
    sample_results = data.get('sample_results', [])
    
    if not sample_results:
        print("❌ No sample results found in pickle file")
        return
    
    # Test with multiple samples
    test_samples = sample_results[:30]  # Test with first 30 samples
    
    print(f"\n📊 Testing with {len(test_samples)} samples...")
    
    improved_predictions = []
    original_predictions = []
    references = []
    sources = []
    
    print("\n📊 Processing samples...")
    print("-" * 60)
    
    for i, sample in enumerate(test_samples):
        print(f"Processing sample {i+1}/{len(test_samples)}...", end="\r")
        
        source_text = sample.get('input', '')
        reference_text = sample.get('reference', '')
        original_pred = sample.get('prediction', '')
        
        if not source_text or not reference_text:
            continue
        
        # Get improved prediction from API
        try:
            response = requests.post(
                'http://localhost:5000/api/simplify',
                json={'text': source_text},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                improved_pred = result.get('simplified', '')
                
                if improved_pred:
                    improved_predictions.append(improved_pred)
                    original_predictions.append(original_pred)
                    references.append(reference_text)
                    sources.append(source_text)
            else:
                print(f"\n⚠️  API Error for sample {i+1}: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                continue
                
        except requests.exceptions.ConnectionError as e:
            print(f"\n❌ CONNECTION ERROR for sample {i+1}")
            print(f"   Error: {e}")
            print("\n⚠️  The server may have stopped. Please:")
            print("   1. Check if the server is still running")
            print("   2. Restart the server: python start_server.py")
            print("   3. Run this test script again")
            print("\n   Stopping tests...")
            break
        except requests.exceptions.Timeout:
            print(f"\n⚠️  Timeout for sample {i+1} (request took >60 seconds)")
            continue
        except requests.exceptions.RequestException as e:
            print(f"\n⚠️  Request error for sample {i+1}: {e}")
            continue
    
    print(f"\n✅ Successfully processed {len(improved_predictions)} samples")
    
    if not improved_predictions:
        print("❌ No samples processed successfully")
        return
    
    # Calculate metrics
    print("\n📊 Calculating Metrics...")
    print("-" * 60)
    
    # Original metrics
    original_sari_scores = []
    original_bleu_scores = []
    original_addition_f1 = []
    original_deletion_f1 = []
    original_keep_f1 = []
    
    # Improved metrics
    improved_sari_scores = []
    improved_bleu_scores = []
    improved_addition_f1 = []
    improved_deletion_f1 = []
    improved_keep_f1 = []
    
    for i in range(len(improved_predictions)):
        # Calculate SARI components for original
        orig_sari_components = calculate_sari_components(
            original_predictions[i], references[i], sources[i]
        )
        orig_sari = orig_sari_components['sari']
        orig_bleu = calculate_bleu(original_predictions[i], references[i])
        
        # Calculate SARI components for improved
        imp_sari_components = calculate_sari_components(
            improved_predictions[i], references[i], sources[i]
        )
        imp_sari = imp_sari_components['sari']
        imp_bleu = calculate_bleu(improved_predictions[i], references[i])
        
        original_sari_scores.append(orig_sari)
        original_bleu_scores.append(orig_bleu)
        original_addition_f1.append(orig_sari_components['addition_f1'])
        original_deletion_f1.append(orig_sari_components['deletion_f1'])
        original_keep_f1.append(orig_sari_components['keep_f1'])
        
        improved_sari_scores.append(imp_sari)
        improved_bleu_scores.append(imp_bleu)
        improved_addition_f1.append(imp_sari_components['addition_f1'])
        improved_deletion_f1.append(imp_sari_components['deletion_f1'])
        improved_keep_f1.append(imp_sari_components['keep_f1'])
    
    # Calculate averages
    avg_original_sari = sum(original_sari_scores) / len(original_sari_scores)
    avg_original_bleu = sum(original_bleu_scores) / len(original_bleu_scores)
    avg_original_addition = sum(original_addition_f1) / len(original_addition_f1)
    avg_original_deletion = sum(original_deletion_f1) / len(original_deletion_f1)
    avg_original_keep = sum(original_keep_f1) / len(original_keep_f1)
    
    avg_improved_sari = sum(improved_sari_scores) / len(improved_sari_scores)
    avg_improved_bleu = sum(improved_bleu_scores) / len(improved_bleu_scores)
    avg_improved_addition = sum(improved_addition_f1) / len(improved_addition_f1)
    avg_improved_deletion = sum(improved_deletion_f1) / len(improved_deletion_f1)
    avg_improved_keep = sum(improved_keep_f1) / len(improved_keep_f1)
    
    # Results
    print("\n🎯 SARI SCORE COMPARISON")
    print("=" * 60)
    print(f"Original SARI Score:  {avg_original_sari:.2f}/100")
    print(f"Improved SARI Score:  {avg_improved_sari:.2f}/100")
    print(f"SARI Improvement:     {avg_improved_sari - avg_original_sari:+.2f} points")
    print(f"SARI Improvement %:   {((avg_improved_sari - avg_original_sari) / avg_original_sari * 100):+.1f}%")
    
    print("\n📈 SARI COMPONENTS BREAKDOWN")
    print("=" * 60)
    print("Addition F1:")
    print(f"  Original:  {avg_original_addition:.3f}")
    print(f"  Improved:  {avg_improved_addition:.3f}")
    print(f"  Change:    {avg_improved_addition - avg_original_addition:+.3f}")
    
    print("\nDeletion F1:")
    print(f"  Original:  {avg_original_deletion:.3f}")
    print(f"  Improved:  {avg_improved_deletion:.3f}")
    print(f"  Change:    {avg_improved_deletion - avg_original_deletion:+.3f}")
    
    print("\nKeep F1:")
    print(f"  Original:  {avg_original_keep:.3f}")
    print(f"  Improved:  {avg_improved_keep:.3f}")
    print(f"  Change:    {avg_improved_keep - avg_original_keep:+.3f}")
    
    print("\n📊 BLEU SCORE COMPARISON")
    print("=" * 60)
    print(f"Original BLEU Score:  {avg_original_bleu:.3f}")
    print(f"Improved BLEU Score:  {avg_improved_bleu:.3f}")
    print(f"BLEU Change:          {avg_improved_bleu - avg_original_bleu:+.3f}")
    
    print("\n📏 LENGTH ANALYSIS")
    print("=" * 60)
    
    # Length analysis
    orig_lengths = [len(tokenize(pred)) for pred in original_predictions]
    imp_lengths = [len(tokenize(pred)) for pred in improved_predictions]
    ref_lengths = [len(tokenize(ref)) for ref in references]
    src_lengths = [len(tokenize(src)) for src in sources]
    
    avg_orig_length = sum(orig_lengths) / len(orig_lengths)
    avg_imp_length = sum(imp_lengths) / len(imp_lengths)
    avg_ref_length = sum(ref_lengths) / len(ref_lengths)
    avg_src_length = sum(src_lengths) / len(src_lengths)
    
    print(f"Average Source Length:    {avg_src_length:.1f} words")
    print(f"Average Reference Length: {avg_ref_length:.1f} words")
    print(f"Average Original Length:  {avg_orig_length:.1f} words")
    print(f"Average Improved Length:  {avg_imp_length:.1f} words")
    print(f"Length Improvement:      {avg_imp_length - avg_orig_length:+.1f} words")
    print(f"Length Ratio (vs ref):    {avg_imp_length/avg_ref_length:.2f}")
    print(f"Length Ratio (vs src):    {avg_imp_length/avg_src_length:.2f}")
    
    print("\n✅ QUALITY ANALYSIS")
    print("=" * 60)
    
    # Artifact analysis
    orig_artifacts = sum(1 for pred in original_predictions 
                       if any(art in pred.lower() for art in ['<extra_id', 'sinhala', 'english', '<pad>', '<eos>']))
    imp_artifacts = sum(1 for pred in improved_predictions 
                       if any(art in pred.lower() for art in ['<extra_id', 'sinhala', 'english', '<pad>', '<eos>']))
    
    print(f"Original Artifacts:       {orig_artifacts}/{len(original_predictions)} samples")
    print(f"Improved Artifacts:       {imp_artifacts}/{len(improved_predictions)} samples")
    print(f"Artifact Reduction:       {orig_artifacts - imp_artifacts} samples")
    
    # Sinhala content analysis
    orig_sinhala_ratios = []
    imp_sinhala_ratios = []
    
    for pred in original_predictions:
        sinhala_chars = sum(1 for char in pred if '\u0D80' <= char <= '\u0DFF')
        total_chars = len(pred)
        ratio = sinhala_chars / max(total_chars, 1)
        orig_sinhala_ratios.append(ratio)
    
    for pred in improved_predictions:
        sinhala_chars = sum(1 for char in pred if '\u0D80' <= char <= '\u0DFF')
        total_chars = len(pred)
        ratio = sinhala_chars / max(total_chars, 1)
        imp_sinhala_ratios.append(ratio)
    
    avg_orig_sinhala = sum(orig_sinhala_ratios) / len(orig_sinhala_ratios)
    avg_imp_sinhala = sum(imp_sinhala_ratios) / len(imp_sinhala_ratios)
    
    print(f"Original Sinhala Ratio:   {avg_orig_sinhala:.3f}")
    print(f"Improved Sinhala Ratio:   {avg_imp_sinhala:.3f}")
    
    print("\n🎉 SUMMARY")
    print("=" * 60)
    
    if avg_improved_sari > avg_original_sari:
        improvement = avg_improved_sari - avg_original_sari
        improvement_pct = (improvement / avg_original_sari * 100) if avg_original_sari > 0 else 0
        print(f"✅ SARI Score IMPROVED!")
        print(f"   From {avg_original_sari:.2f} to {avg_improved_sari:.2f}")
        print(f"   Improvement: {improvement:+.2f} points ({improvement_pct:+.1f}%)")
        
        if avg_improved_sari >= 30:
            print("   🎉 EXCELLENT! Achieved 30+ SARI score!")
        elif avg_improved_sari >= 25:
            print("   👍 GOOD! Achieved 25+ SARI score!")
        elif avg_improved_sari >= 20:
            print("   📈 FAIR! Achieved 20+ SARI score!")
        else:
            print("   ⚠️  Below 20, but improved from baseline")
    else:
        print("⚠️  SARI Score needs more work")
        print(f"   Current: {avg_improved_sari:.2f}")
    
    if avg_imp_length > avg_orig_length:
        print(f"✅ Output Length IMPROVED!")
        print(f"   From {avg_orig_length:.1f} to {avg_imp_length:.1f} words")
    
    if imp_artifacts < orig_artifacts:
        print(f"✅ Artifacts REDUCED!")
        print(f"   From {orig_artifacts} to {imp_artifacts} samples")
    
    # Component improvements
    print("\n📊 Component Improvements:")
    if avg_improved_addition > avg_original_addition:
        print(f"  ✅ Addition F1: {avg_improved_addition - avg_original_addition:+.3f}")
    if avg_improved_deletion > avg_original_deletion:
        print(f"  ✅ Deletion F1: {avg_improved_deletion - avg_original_deletion:+.3f}")
    if avg_improved_keep > avg_original_keep:
        print(f"  ✅ Keep F1: {avg_improved_keep - avg_original_keep:+.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Testing completed successfully!")
    print("=" * 60)


def calculate_sari_components(prediction, reference, source):
    """Calculate SARI components"""
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
    
    def f1_score(pred_set, ref_set):
        if not pred_set and not ref_set:
            return 1.0
        if not pred_set or not ref_set:
            return 0.0
        
        precision = len(pred_set & ref_set) / len(pred_set) if pred_set else 0.0
        recall = len(pred_set & ref_set) / len(ref_set) if ref_set else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    addition_f1 = f1_score(addition_pred, addition_ref)
    deletion_f1 = f1_score(deletion_pred, deletion_ref)
    keep_f1 = f1_score(keep_pred, keep_ref)
    
    sari = (addition_f1 + deletion_f1 + keep_f1) / 3
    
    return {
        'addition_f1': addition_f1,
        'deletion_f1': deletion_f1,
        'keep_f1': keep_f1,
        'sari': sari * 100
    }


if __name__ == "__main__":
    test_sari_improvements()
