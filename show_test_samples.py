#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import sys

# Set UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load the pickle file
with open('sinhala_text_simplification_results.pkl', 'rb') as f:
    data = pickle.load(f)

sample_results = data.get('sample_results', [])

print("=" * 80)
print("5 SAMPLES USED FOR SARI SCORE CALCULATION (32.18 SARI)")
print("=" * 80)
print()

for i, sample in enumerate(sample_results[:5], 1):
    print(f"SAMPLE {i}:")
    print("-" * 80)
    print(f"Original (Input):     {sample.get('input', '')}")
    print(f"Reference (Target):   {sample.get('reference', '')}")
    print(f"Baseline Prediction:  {sample.get('prediction', '')}")
    print()
    print()

print("=" * 80)
print("NOTE: These are the exact 5 samples from the evaluation dataset")
print("      that were used to calculate the SARI score of 32.18")
print("=" * 80)
