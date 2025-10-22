"""
Verify grid dimensions for all split parquet files
"""
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_grid(parquet_file):
    """Analyze grid dimensions for a parquet file"""
    df = pd.read_parquet(parquet_file)
    
    # Get one layer/step/variant
    subset = df[(df['layer']==0) & (df['generation_step']==0) & (df['variant']=='Full')]
    
    # Get vision tokens
    vision = subset[subset['token_masked'].str.contains('image_pad', case=False, na=False)]
    
    num_tokens = len(vision['token_position'].unique())
    sqrt_val = np.sqrt(num_tokens)
    
    # Find best rectangular grid
    factors = []
    for i in range(1, int(sqrt_val) + 5):
        if num_tokens % i == 0:
            j = num_tokens // i
            factors.append((j, i))  # width, height
    
    # Find closest to square
    best = min(factors, key=lambda x: abs(x[0] - x[1]))
    
    return {
        'tokens': num_tokens,
        'sqrt': sqrt_val,
        'is_square': sqrt_val == int(sqrt_val),
        'best_grid': best,
        'all_grids': factors
    }

# Find all split parquet files
output_dir = Path('output')
split_files = [
    'eras_text_orb_looking_at_eras_text.parquet',
    'eras_text_orb_looking_at_eras_figures.parquet', 
    'eras_text_orb_looking_at_eras_legs.parquet',
    'eras_text_orb_looking_at_orb_looking_at.parquet'
]

print("=" * 70)
print("GRID DIMENSION VERIFICATION")
print("=" * 70)

for filename in split_files:
    filepath = output_dir / filename
    if not filepath.exists():
        print(f"\n{filename}: FILE NOT FOUND")
        continue
    
    prompt_name = filename.replace('eras_text_orb_looking_at_', '').replace('.parquet', '')
    
    print(f"\n{prompt_name}:")
    result = analyze_grid(filepath)
    
    print(f"  Vision tokens: {result['tokens']}")
    print(f"  Square root: {result['sqrt']:.4f}")
    print(f"  Is perfect square: {result['is_square']}")
    print(f"  Best grid: {result['best_grid'][0]}x{result['best_grid'][1]}")
    print(f"  All possible grids: {', '.join(f'{w}x{h}' for w, h in result['all_grids'])}")
    
    # Check if it matches expected
    if prompt_name == 'orb_looking_at':
        expected = (24, 24)
        match = result['best_grid'] == expected
        print(f"  Expected: 24x24 - {'[OK] MATCH' if match else '[X] MISMATCH'}")
    elif 'eras' in prompt_name:
        expected = (24, 22)
        match = result['best_grid'] == expected
        print(f"  Expected: 24x22 - {'[OK] MATCH' if match else '[X] MISMATCH'}")

print("\n" + "=" * 70)

