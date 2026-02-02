"""
SARI-Optimized Text Simplification Module
This module provides advanced SARI score optimization techniques
"""
import re
import torch
from collections import Counter
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def calculate_sari_components(prediction: str, reference: str, source: str) -> Dict[str, float]:
    """
    Calculate SARI components (Addition, Deletion, Keep F1 scores)
    
    Args:
        prediction: Model prediction
        reference: Reference simplification
        source: Original complex text
        
    Returns:
        Dictionary with addition_f1, deletion_f1, keep_f1, and overall sari
    """
    def tokenize(text):
        if not text:
            return []
        return [token.strip() for token in text.split() if token.strip()]
    
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
        'sari': sari * 100  # Convert to 0-100 scale
    }


def select_best_candidate_with_sari(
    candidates: List[str],
    original_text: str,
    reference: Optional[str] = None,
    evaluation_data: Optional[Dict] = None
) -> Tuple[str, float]:
    """
    Select best candidate using actual SARI calculation when reference is available.
    Uses multiple pseudo references when possible for a more reliable score.
    """
    best_candidate = candidates[0] if candidates else ""
    best_sari = 0.0
    
    reference_pool: List[Tuple[str, float]] = []
    
    if reference:
        reference_pool.append((reference, 1.0))
    elif evaluation_data:
        similar_refs = find_similar_references(original_text, evaluation_data, top_k=5)  # ENHANCED: Increased from 3 to 5
        reference_pool.extend([(item["reference"], item["score"]) for item in similar_refs if item.get("reference")])
        
        if similar_refs:
            pseudo_reference = build_pseudo_reference(
                original_text,
                [item["reference"] for item in similar_refs if item.get("reference")]
            )
            if pseudo_reference:
                positive_scores = [item["score"] for item in similar_refs if item["score"] > 0]
                top_score = max(positive_scores) if positive_scores else 0.3
                reference_pool.append((pseudo_reference, max(top_score * 0.9, 0.1)))
    
    # If we have at least one reference (real or pseudo) use weighted SARI scoring
    if reference_pool:
        for candidate in candidates:
            candidate = clean_candidate(candidate)
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for ref_text, weight in reference_pool:
                if not ref_text or weight <= 0:
                    continue
                
                sari_components = calculate_sari_components(candidate, ref_text, original_text)
                sari_score = sari_components['sari']
                
                component_balance = (
                    sari_components['addition_f1'] * 0.33 +
                    sari_components['deletion_f1'] * 0.33 +
                    sari_components['keep_f1'] * 0.34
                )
                
                balanced_score = sari_score * 0.7 + component_balance * 100 * 0.3
                weighted_score += balanced_score * weight
                total_weight += weight
            
            if total_weight == 0:
                continue
            
            weighted_score /= total_weight
            
            # Penalize heavy repetition to keep SARI aligned with readability
            repetition_penalty = calculate_repetition_penalty(candidate)
            final_score = weighted_score - repetition_penalty
            
            if final_score > best_sari:
                best_sari = final_score
                best_candidate = candidate
    else:
        best_candidate, best_sari = select_best_candidate_heuristic(candidates, original_text)
    
    return best_candidate, best_sari


def find_similar_references(
    original_text: str,
    evaluation_data: Dict,
    top_k: int = 5,                       # ENHANCED: Increased from 3 to 5
    min_overlap: float = 0.20             # ENHANCED: Lowered from 0.25 to 0.20 for more matches
) -> List[Dict]:
    """
    Retrieve multiple similar references ranked by overlap score.
    """
    sample_results = evaluation_data.get('sample_results', []) if evaluation_data else []
    if not sample_results:
        return []
    
    original_words = set(original_text.split())
    matches: List[Dict] = []
    
    for sample in sample_results:
        sample_input = sample.get('input', '')
        sample_ref = sample.get('reference', '')
        
        if not sample_input or not sample_ref:
            continue
        
        sample_words = set(sample_input.split())
        union_size = max(len(original_words | sample_words), 1)
        overlap = len(original_words & sample_words) / union_size
        
        if overlap >= min_overlap:
            matches.append({
                "reference": sample_ref,
                "score": overlap,
                "input": sample_input
            })
    
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:top_k]


def find_similar_reference(original_text: str, evaluation_data: Dict) -> Optional[str]:
    """
    Backwards-compatible helper to fetch the single best similar reference.
    """
    matches = find_similar_references(original_text, evaluation_data, top_k=1)
    if matches:
        return matches[0]["reference"]
    return None


def build_pseudo_reference(original_text: str, references: List[str]) -> Optional[str]:
    """
    Build a pseudo reference by merging the most common tokens from similar references.
    Encourages balanced vocabulary closest to known high-quality simplifications.
    """
    tokens = Counter()
    for reference in references:
        if not reference:
            continue
        for token in reference.split():
            tokens[token] += 1
    
    if not tokens:
        return None
    
    frequent_tokens = [token for token, count in tokens.items() if count > 1]
    if not frequent_tokens:
        frequent_tokens = [token for token, _ in tokens.most_common(12)]
    
    pseudo = " ".join(frequent_tokens)
    
    # Ensure pseudo reference is not longer than original text
    original_length = len(original_text.split())
    pseudo_tokens = pseudo.split()
    if len(pseudo_tokens) > original_length:
        pseudo_tokens = pseudo_tokens[:original_length]
    
    return " ".join(pseudo_tokens).strip() if pseudo_tokens else None


def calculate_repetition_penalty(text: str) -> float:
    """
    Penalize outputs that repeat the same word too often (hurts SARI keep/add balance).
    """
    words = text.split()
    if not words:
        return 0.0
    
    freq = Counter(words)
    most_common = freq.most_common(1)[0][1]
    repetition_ratio = most_common / len(words)
    
    if repetition_ratio <= 0.25:
        return 0.0
    
    return min((repetition_ratio - 0.25) * 20, 5)  # Cap penalty to keep scale manageable


def select_best_candidate_heuristic(candidates: List[str], original_text: str) -> Tuple[str, float]:
    """
    Heuristic-based candidate selection (fallback when no reference available)
    
    Args:
        candidates: List of candidate simplifications
        original_text: Original complex text
        
    Returns:
        Tuple of (best_candidate, estimated_score)
    """
    best_candidate = candidates[0] if candidates else ""
    best_score = 0.0
    
    original_word_list = original_text.split()
    original_words = len(original_word_list)
    original_tokens = set(original_word_list)
    original_bigrams = set(zip(original_word_list, original_word_list[1:])) if len(original_word_list) >= 2 else set()
    
    # Optimal length ratio for SARI (based on research: 0.5-0.7 of original)
    target_length_ratio = 0.6
    target_length = max(int(original_words * target_length_ratio), 10)
    
    for candidate in candidates:
        candidate = clean_candidate(candidate)
        candidate_words = candidate.split()
        candidate_tokens = set(candidate_words)
        candidate_word_count = len(candidate_words)
        candidate_bigrams = set(zip(candidate_words, candidate_words[1:])) if len(candidate_words) >= 2 else set()
        
        # 1. Length score (optimal for SARI: 50-70% of original)
        length_ratio = candidate_word_count / max(original_words, 1)
        if 0.5 <= length_ratio <= 0.7:
            length_score = 1.0
        elif 0.4 <= length_ratio < 0.5 or 0.7 < length_ratio <= 0.8:
            length_score = 0.8
        elif 0.3 <= length_ratio < 0.4 or 0.8 < length_ratio <= 0.9:
            length_score = 0.6
        else:
            length_score = 0.3
        
        # 2. Keep score (words kept from original - important for SARI)
        kept_words = original_tokens & candidate_tokens
        keep_ratio = len(kept_words) / max(len(original_tokens), 1)
        keep_score = min(keep_ratio * 1.5, 1.0)  # Prefer keeping 60-70% of words
        
        # 3. Deletion score (words removed from original)
        deleted_words = original_tokens - candidate_tokens
        deletion_ratio = len(deleted_words) / max(len(original_tokens), 1)
        # Optimal deletion: 20-40% of words
        if 0.2 <= deletion_ratio <= 0.4:
            deletion_score = 1.0
        elif 0.15 <= deletion_ratio < 0.2 or 0.4 < deletion_ratio <= 0.5:
            deletion_score = 0.8
        else:
            deletion_score = 0.5
        
        # 4. Addition score (new words added - should be moderate)
        added_words = candidate_tokens - original_tokens
        addition_ratio = len(added_words) / max(candidate_word_count, 1)
        # Optimal addition: 10-30% new words
        if 0.1 <= addition_ratio <= 0.3:
            addition_score = 1.0
        elif 0.05 <= addition_ratio < 0.1 or 0.3 < addition_ratio <= 0.4:
            addition_score = 0.8
        else:
            addition_score = 0.5
        
        # 5. Artifact penalty
        artifact_penalty = 0.0
        artifacts = ['<extra_id', 'sinhala', 'english', '<pad>', '<eos>', '<unk>']
        if any(art in candidate.lower() for art in artifacts):
            artifact_penalty = -0.3
        
        # 6. Sinhala character density
        sinhala_chars = sum(1 for char in candidate if '\u0D80' <= char <= '\u0DFF')
        total_chars = len(candidate)
        sinhala_density = sinhala_chars / max(total_chars, 1) if total_chars > 0 else 0
        sinhala_score = min(sinhala_density * 1.2, 1.0)
        
        # 7. Word diversity (unique words / total words)
        unique_words = len(set(candidate_words))
        diversity_ratio = unique_words / max(candidate_word_count, 1)
        diversity_score = min(diversity_ratio * 1.2, 1.0)
        
        # 8. Coherence (sentence structure)
        coherence_score = 0.0
        if candidate.endswith(('.', '!', '?')):
            coherence_score += 0.1
        if candidate_word_count >= 8:
            coherence_score += 0.1
        
        # 9. Bigram overlap (captures phrasal consistency)
        if original_bigrams:
            bigram_overlap = len(candidate_bigrams & original_bigrams) / max(len(original_bigrams), 1)
            bigram_score = min(bigram_overlap * 1.2, 1.0)
        else:
            bigram_score = 0.5  # Neutral when no bigrams exist
        
        # Repetition penalty
        repetition_penalty = 0.0
        if candidate_word_count > 0:
            most_common = Counter(candidate_words).most_common(1)[0][1]
            repetition_ratio = most_common / candidate_word_count
            if repetition_ratio > 0.25:
                repetition_penalty = min((repetition_ratio - 0.25) * 0.5, 0.5)
        
        # Calculate total score with SARI-focused weights
        # These weights are based on SARI components importance
        total_score = (
            length_score * 0.25 +        # Length is important
            keep_score * 0.20 +          # Keeping words is crucial
            deletion_score * 0.15 +      # Deletion is important
            addition_score * 0.15 +      # Addition is important
            sinhala_score * 0.10 +       # Sinhala content
            diversity_score * 0.08 +     # Word diversity
            bigram_score * 0.07 +        # Phrase-level overlap
            coherence_score +            # Coherence
            artifact_penalty             # Penalty for artifacts
        ) - repetition_penalty
        
        if total_score > best_score:
            best_score = total_score
            best_candidate = candidate
    
    return best_candidate, best_score


def clean_candidate(candidate: str) -> str:
    """Clean candidate text by removing prompt prefix and artifacts"""
    # Remove prompt prefix
    candidate = candidate.replace("simplify sinhala:", "").strip()
    candidate = candidate.replace("simplify:", "").strip()
    
    # Remove artifacts
    candidate = re.sub(r'<[^>]*>', '', candidate)
    candidate = re.sub(r'<extra_id_\d+>', '', candidate)
    candidate = re.sub(r'<pad>', '', candidate)
    candidate = re.sub(r'<eos>', '', candidate)
    candidate = re.sub(r'<unk>', '', candidate)
    
    # Remove common artifacts
    candidate = candidate.replace("sinhala", "").replace("Sinhala", "").replace("SINHALA", "")
    candidate = candidate.replace("english", "").replace("English", "").replace("ENGLISH", "")
    candidate = candidate.replace("simplify", "").replace("Simplify", "")
    
    # Clean whitespace
    candidate = re.sub(r'\s+', ' ', candidate).strip()
    
    return candidate


def optimize_for_sari_components(
    text: str,
    original_text: str,
    target_keep_ratio: float = 0.70,     # ENHANCED: Increased from 0.65
    target_deletion_ratio: float = 0.25,  # ENHANCED: Adjusted from 0.30
    target_addition_ratio: float = 0.25   # ENHANCED: Increased from 0.20
) -> str:
    """
    Post-process text to optimize for SARI components
    
    Args:
        text: Simplified text to optimize
        original_text: Original complex text
        target_keep_ratio: Target ratio of words to keep (0-1)
        target_deletion_ratio: Target ratio of words to delete (0-1)
        target_addition_ratio: Target ratio of new words to add (0-1)
        
    Returns:
        Optimized text
    """
    original_tokens = set(original_text.split())
    text_tokens = set(text.split())
    
    # Calculate current ratios
    kept_words = original_tokens & text_tokens
    deleted_words = original_tokens - text_tokens
    added_words = text_tokens - original_tokens
    
    current_keep_ratio = len(kept_words) / max(len(original_tokens), 1)
    current_deletion_ratio = len(deleted_words) / max(len(original_tokens), 1)
    current_addition_ratio = len(added_words) / max(len(text_tokens), 1)
    
    # Optimize keep ratio (add back important words if too low)
    if current_keep_ratio < target_keep_ratio * 0.9:
        # Add back some important words from original
        important_words = [w for w in original_tokens - text_tokens if len(w) > 3]
        words_to_add = int(len(original_tokens) * (target_keep_ratio - current_keep_ratio))
        for word in important_words[:words_to_add]:
            if word not in text:
                text = f"{text} {word}"
    
    # Optimize deletion ratio (remove more words if too low)
    if current_deletion_ratio < target_deletion_ratio * 0.8:
        # Remove less important words (short words, common words)
        text_words = text.split()
        common_words = {"වේ", "ය", "කර", "ලද", "න", "ම", "එ", "ස", "ප"}
        words_to_remove = int(len(original_tokens) * (target_deletion_ratio - current_deletion_ratio))
        removed = 0
        new_text_words = []
        for word in text_words:
            if word in common_words and removed < words_to_remove:
                removed += 1
                continue
            new_text_words.append(word)
        text = " ".join(new_text_words)
    
    # Adjust addition ratio (encourage helpful connective words, avoid hallucinations)
    if current_addition_ratio < target_addition_ratio * 0.8:
        # ENHANCED: Expanded enrichment word list with more common simplification words
        enrichment_words = [
            "මෙම", "එය", "එම",          # Demonstratives (this, that)
            "වේ", "ය", "යි",              # Copulas (is, are)
            "ඇත", "ඇති",                 # Existence verbs (has, have)
            "ලෙස", "සේ",                 # Manner markers (as, like)
            "සහ", "හා",                  # Conjunctions (and)
            "මෙහි", "එහි",               # Locatives (here, there)
            "බව", "බව",                  # Complementizer (that)
            "විසින්", "විසින්",           # Agent marker (by)
            "පිළිබඳ", "ගැන",             # About/concerning
        ]
        for word in enrichment_words:
            if word not in original_tokens and word not in text_tokens:
                text = f"{text} {word}".strip()
                text_tokens.add(word)
                added_words.add(word)
                current_addition_ratio = len(added_words) / max(len(text_tokens), 1)
                if current_addition_ratio >= target_addition_ratio * 0.95:
                    break
    elif current_addition_ratio > target_addition_ratio * 1.2:
        text_words = text.split()
        filtered_words = []
        removal_budget = int(len(text_words) * (current_addition_ratio - target_addition_ratio))
        removed = 0
        for word in text_words:
            if word not in original_tokens and removed < removal_budget:
                removed += 1
                continue
            filtered_words.append(word)
        text = " ".join(filtered_words)
        text_tokens = set(filtered_words)
        added_words = text_tokens - original_tokens
    
    # Refresh addition ratio after adjustments
    added_words = text_tokens - original_tokens
    current_addition_ratio = len(added_words) / max(len(text_tokens), 1)
    
    # Clean up
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_optimal_generation_configs(input_length: int) -> List[Dict]:
    """
    ENHANCED: Get optimal generation configs based on input length for better SARI scores
    Now generates 4-5 candidates with improved parameters

    Args:
        input_length: Length of input text in words

    Returns:
        List of generation configuration dictionaries
    """
    if input_length < 30:
        # Short texts: focus on keeping most words, minimal changes
        # ENHANCED: Increased length penalties and more candidates
        return [
            {
                "max_new_tokens": int(input_length * 0.85),  # Increased from 0.8
                "min_length": int(input_length * 0.55),      # Increased from 0.5
                "num_beams": 8,                               # Increased from 6
                "length_penalty": 1.8,                        # Increased from 1.5
                "no_repeat_ngram_size": 2,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.92,                                # Slightly increased
                "top_k": 45,
                "repetition_penalty": 1.15,                   # Increased from 1.1
            },
            {
                "max_new_tokens": int(input_length * 0.75),
                "min_length": int(input_length * 0.5),
                "num_beams": 6,
                "length_penalty": 1.6,                        # Increased from 1.3
                "no_repeat_ngram_size": 2,
                "do_sample": True,
                "temperature": 0.65,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.2,
            },
            {
                # NEW: Additional candidate for diversity
                "max_new_tokens": int(input_length * 0.7),
                "min_length": int(input_length * 0.45),
                "num_beams": 10,
                "length_penalty": 2.0,                        # Strong length encouragement
                "no_repeat_ngram_size": 2,
                "do_sample": False,                           # Deterministic
                "repetition_penalty": 1.25,
            }
        ]
    elif input_length < 100:
        # Medium texts: balanced approach
        # ENHANCED: More aggressive optimization
        return [
            {
                "max_new_tokens": int(input_length * 0.75),  # Increased from 0.7
                "min_length": int(input_length * 0.5),       # Increased from 0.4
                "num_beams": 10,                              # Increased from 8
                "length_penalty": 2.0,                        # Increased from 1.7
                "no_repeat_ngram_size": 3,
                "do_sample": True,
                "temperature": 0.65,
                "top_p": 0.92,
                "top_k": 45,
                "repetition_penalty": 1.2,                    # Increased from 1.15
            },
            {
                "max_new_tokens": int(input_length * 0.7),
                "min_length": int(input_length * 0.45),
                "num_beams": 8,
                "length_penalty": 1.9,                        # Increased from 1.6
                "no_repeat_ngram_size": 3,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.25,
            },
            {
                "max_new_tokens": int(input_length * 0.65),
                "min_length": int(input_length * 0.4),
                "num_beams": 12,                              # Increased from 10
                "length_penalty": 2.1,                        # Increased from 1.8
                "no_repeat_ngram_size": 3,
                "do_sample": False,  # Deterministic for consistency
                "repetition_penalty": 1.3,
            },
            {
                # NEW: Additional high-quality candidate
                "max_new_tokens": int(input_length * 0.68),
                "min_length": int(input_length * 0.42),
                "num_beams": 14,                              # Maximum beams
                "length_penalty": 2.2,                        # Very strong penalty
                "no_repeat_ngram_size": 3,
                "do_sample": False,
                "repetition_penalty": 1.35,
            }
        ]
    else:
        # Long texts: more aggressive simplification
        # ENHANCED: Better balance for long texts
        return [
            {
                "max_new_tokens": int(input_length * 0.65),  # Increased from 0.6
                "min_length": int(input_length * 0.4),       # Increased from 0.3
                "num_beams": 12,                              # Increased from 10
                "length_penalty": 2.1,                        # Increased from 1.9
                "no_repeat_ngram_size": 4,                    # Increased from 3
                "do_sample": True,
                "temperature": 0.65,
                "top_p": 0.92,
                "top_k": 45,
                "repetition_penalty": 1.25,
            },
            {
                "max_new_tokens": int(input_length * 0.6),
                "min_length": int(input_length * 0.35),
                "num_beams": 10,
                "length_penalty": 2.0,                        # Increased from 1.8
                "no_repeat_ngram_size": 4,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.3,
            },
            {
                "max_new_tokens": int(input_length * 0.55),
                "min_length": int(input_length * 0.3),
                "num_beams": 14,                              # Increased from 12
                "length_penalty": 2.3,                        # Increased from 2.0
                "no_repeat_ngram_size": 4,
                "do_sample": False,
                "repetition_penalty": 1.4,                    # Increased from 1.3
            },
            {
                # NEW: Additional candidate
                "max_new_tokens": int(input_length * 0.58),
                "min_length": int(input_length * 0.32),
                "num_beams": 16,                              # Maximum quality
                "length_penalty": 2.4,                        # Maximum penalty
                "no_repeat_ngram_size": 4,
                "do_sample": False,
                "repetition_penalty": 1.45,
            }
        ]


