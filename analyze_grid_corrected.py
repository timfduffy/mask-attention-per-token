"""
Corrected analysis of vision token layout with gaps
"""
import pandas as pd
from pathlib import Path

# Load the parquet file
parquet_file = Path('output/eras_text_orb_looking_at.parquet')
df = pd.read_parquet(parquet_file)

# Look at one layer/step combination
layer_0 = df[df['layer'] == 0]
step_0 = layer_0[layer_0['generation_step'] == 0]

# Separate vision tokens from text tokens  
vision_tokens = step_0[step_0['token_masked'].str.contains('vision_token|image_pad|img_', case=False, na=False)]

print("=== VISION TOKEN RANGE ANALYSIS ===\n")

# Get all vision token positions
vision_positions = sorted(vision_tokens['token_position'].unique())

print(f"Total vision tokens: {len(vision_positions)}")
print(f"Position range: {min(vision_positions)} to {max(vision_positions)}\n")

# Find contiguous ranges
ranges = []
start = vision_positions[0]
prev = start

for pos in vision_positions[1:]:
    if pos != prev + 1:
        # Gap detected - close current range
        ranges.append((start, prev))
        start = pos
    prev = pos
ranges.append((start, prev))

print(f"Vision tokens are split into {len(ranges)} contiguous range(s):\n")
for i, (start, end) in enumerate(ranges, 1):
    count = end - start + 1
    print(f"  Range {i}: positions {start} to {end} = {count} tokens")

print(f"\nTotal across all ranges: {sum(end - start + 1 for start, end in ranges)}")

# Calculate actual grid layout
total_vision_tokens = len(vision_positions)
import math
sqrt_val = math.sqrt(total_vision_tokens)

if sqrt_val == int(sqrt_val):
    grid_size = int(sqrt_val)
    print(f"\n=== GRID LAYOUT ===")
    print(f"{total_vision_tokens} tokens = {grid_size}x{grid_size} (perfect square)")
else:
    print(f"\n=== GRID LAYOUT ===")
    print(f"{total_vision_tokens} tokens is NOT a perfect square!")
    print(f"Square root: {sqrt_val:.4f}")
    
    # Find best rectangular grid
    factors = []
    for i in range(1, int(sqrt_val) + 2):
        if total_vision_tokens % i == 0:
            j = total_vision_tokens // i
            factors.append((i, j))
    
    if factors:
        best = min(factors, key=lambda x: abs(x[0] - x[1]))
        print(f"Best rectangular grid: {best[0]}x{best[1]}")
        print(f"\nAll possible grids:")
        for i, j in factors:
            print(f"  {i}x{j}")

# Show what's in the gaps
if len(ranges) > 1:
    print(f"\n=== GAPS BETWEEN VISION TOKEN RANGES ===\n")
    for i in range(len(ranges) - 1):
        gap_start = ranges[i][1] + 1
        gap_end = ranges[i+1][0] - 1
        gap_size = gap_end - gap_start + 1
        
        print(f"Gap {i+1}: positions {gap_start} to {gap_end} ({gap_size} positions)")
        
        # Show tokens in this gap
        gap_tokens = step_0[step_0['token_position'].isin(range(gap_start, gap_end + 1))]
        for _, row in gap_tokens.iterrows():
            print(f"  Pos {row['token_position']:3d}: '{row['token_masked']}'")

print(f"\n=== CONCLUSION ===")
print(f"Your data has {total_vision_tokens} vision tokens")
if sqrt_val == int(sqrt_val):
    print(f"This IS a {int(sqrt_val)}x{int(sqrt_val)} grid as expected!")
else:
    print(f"This is NOT a perfect square grid!")
print(f"\nThe visualization tool filters by token content, so it should")
print(f"correctly identify and display the {total_vision_tokens} vision tokens")
print(f"regardless of their position numbers.")

