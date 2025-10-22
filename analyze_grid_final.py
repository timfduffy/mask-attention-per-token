"""
Final corrected analysis - properly detect gaps in vision token positions
"""
import pandas as pd
from pathlib import Path
import math

# Load the parquet file
parquet_file = Path('output/eras_text_orb_looking_at.parquet')
df = pd.read_parquet(parquet_file)

# Look at one layer/step combination  
layer_0 = df[df['layer'] == 0]
step_0 = layer_0[layer_0['generation_step'] == 0]

# Separate vision tokens from text tokens
vision_tokens = step_0[step_0['token_masked'].str.contains('vision_token|image_pad|img_', case=False, na=False)]
text_tokens = step_0[~step_0['token_masked'].str.contains('vision_token|image_pad|img_', case=False, na=False)]

print("=== VISION TOKEN ANALYSIS ===\n")

# Get all vision and text token positions
vision_positions = set(vision_tokens['token_position'].unique())
text_positions = set(text_tokens['token_position'].unique())

print(f"Total vision tokens: {len(vision_positions)}")
print(f"Total text tokens: {len(text_positions)}")

# Find the min/max range
min_pos = min(vision_positions)
max_pos = max(vision_positions)

print(f"\nVision tokens span positions {min_pos} to {max_pos}")

# Check for text tokens embedded in this range
embedded_text = []
for pos in range(min_pos, max_pos + 1):
    if pos in text_positions:
        embedded_text.append(pos)

if embedded_text:
    print(f"\nWARNING: {len(embedded_text)} TEXT tokens embedded in vision range!")
    print(f"Text token positions: {embedded_text[0]} to {embedded_text[-1]}")
    
    # Find contiguous vision ranges
    ranges = []
    in_range = False
    start = None
    
    for pos in range(min_pos, max_pos + 1):
        if pos in vision_positions:
            if not in_range:
                start = pos
                in_range = True
        else:
            if in_range:
                ranges.append((start, pos - 1))
                in_range = False
    
    if in_range:
        ranges.append((start, max_pos))
    
    print(f"\nVision tokens split into {len(ranges)} contiguous ranges:")
    total = 0
    for i, (start, end) in enumerate(ranges, 1):
        count = end - start + 1
        total += count
        print(f"  Range {i}: positions {start}-{end} ({count} tokens)")
    
    print(f"\nTotal: {total} vision tokens")
    
    # Show what's between ranges
    for i in range(len(ranges) - 1):
        gap_start = ranges[i][1] + 1
        gap_end = ranges[i+1][0] - 1
        print(f"\nBetween Range {i+1} and Range {i+2} (positions {gap_start}-{gap_end}):")
        
        gap_data = step_0[step_0['token_position'].isin(range(gap_start, min(gap_end + 1, gap_start + 20)))]
        for _, row in gap_data.head(10).iterrows():
            print(f"  Pos {row['token_position']:3d}: {repr(row['token_masked'])}")
        
        if gap_end - gap_start > 10:
            print(f"  ... ({gap_end - gap_start - 9} more)")
else:
    print(f"\nVision tokens are CONTIGUOUS (no embedded text tokens)")

# Grid dimensions
total_vision = len(vision_positions)
sqrt_val = math.sqrt(total_vision)

print(f"\n=== GRID DIMENSIONS ===")
print(f"Total vision tokens: {total_vision}")
print(f"Square root: {sqrt_val:.4f}")

if sqrt_val == int(sqrt_val):
    grid_size = int(sqrt_val)
    print(f"Grid: {grid_size}x{grid_size} (PERFECT SQUARE)")
    
    if embedded_text:
        print(f"\nNote: Even though there are embedded text tokens,")
        print(f"the visualization should correctly display the {total_vision} vision tokens")
        print(f"in a {grid_size}x{grid_size} grid by filtering on token content.")
else:
    print(f"Grid: NOT a perfect square")
    
    # Find closest rectangular
    factors = []
    for i in range(1, int(sqrt_val) + 2):
        if total_vision % i == 0:
            j = total_vision // i
            factors.append((i, j))
    
    if factors:
        best = min(factors, key=lambda x: abs(x[0] - x[1]))
        print(f"Best rectangular: {best[0]}x{best[1]}")

# Summary
print(f"\n=== SUMMARY ===")
print(f"Position 549 is a {('VISION' if 549 in vision_positions else 'TEXT')} token")
if 549 not in vision_positions:
    token_549 = step_0[step_0['token_position'] == 549]['token_masked'].iloc[0]
    print(f"  Content at position 549: {repr(token_549)}")
    print(f"  This explains why you thought the highest was 549 -")
    print(f"  you were looking at a position number that's NOT a vision token!")

