"""
Simple analysis script to explore the masking results.
Run this after generating masking_results.csv
"""

import pandas as pd
import numpy as np

def analyze_results(csv_file='masking_results.csv'):
    """Load and provide basic analysis of results"""
    
    print("Loading results...")
    df = pd.read_csv(csv_file)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    print("\n=== Variants ===")
    print(df['variant'].value_counts())
    
    print("\n=== Tokens Analyzed ===")
    print(df['token_masked'].value_counts())
    
    print("\n=== Top 10 Most Impactful Token-Layer-Variant Combinations (by L2) ===")
    top_l2 = df.nlargest(10, 'l2_distance')[['layer', 'token_masked', 'token_position', 'variant', 'l2_distance', 'cosine_distance']]
    print(top_l2.to_string())
    
    print("\n=== Top 10 Most Impactful Token-Layer-Variant Combinations (by Cosine) ===")
    top_cos = df.nlargest(10, 'cosine_distance')[['layer', 'token_masked', 'token_position', 'variant', 'l2_distance', 'cosine_distance']]
    print(top_cos.to_string())
    
    # Analysis by variant
    print("\n=== Average Impact by Variant ===")
    variant_stats = df.groupby('variant')[['l2_distance', 'cosine_distance']].mean().sort_values('l2_distance', ascending=False)
    print(variant_stats)
    
    # Analysis by layer
    print("\n=== Average Impact by Layer (Full variant only) ===")
    layer_stats = df[df['variant'] == 'Full'].groupby('layer')[['l2_distance', 'cosine_distance']].mean()
    print(layer_stats)
    
    # Analysis by token
    print("\n=== Average Impact by Token Masked (Full variant only) ===")
    token_stats = df[df['variant'] == 'Full'].groupby('token_masked')[['l2_distance', 'cosine_distance']].mean().sort_values('l2_distance', ascending=False)
    print(token_stats)
    
    # Find most important heads
    head_variants = df[df['variant'].str.startswith('Head_')]
    if not head_variants.empty:
        print("\n=== Top 10 Most Impactful Heads (across all layers and tokens) ===")
        head_stats = head_variants.groupby('variant')[['l2_distance', 'cosine_distance']].mean().sort_values('l2_distance', ascending=False).head(10)
        print(head_stats)
    
    return df

if __name__ == '__main__':
    df = analyze_results()
    
    print("\n" + "="*70)
    print("Analysis complete! The dataframe is available as 'df' for further exploration.")
    print("="*70)

