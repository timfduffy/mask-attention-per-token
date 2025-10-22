"""
Analyze vision token grid layout from parquet file
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the parquet file
parquet_file = Path('output/eras_text_orb_looking_at.parquet')
df = pd.read_parquet(parquet_file)

print(f"Total rows: {len(df):,}")
print(f"\nColumns: {df.columns.tolist()}")

# Look at one layer/step combination
layer_0 = df[df['layer'] == 0]
step_0 = layer_0[layer_0['generation_step'] == 0]

print(f"\nRows for layer=0, step=0: {len(step_0):,}")

# Separate vision tokens from text tokens
vision_tokens = step_0[step_0['token_masked'].str.contains('vision_token|image_pad|img_', case=False, na=False)]
text_tokens = step_0[~step_0['token_masked'].str.contains('vision_token|image_pad|img_', case=False, na=False)]

print(f"\n=== TOKEN BREAKDOWN ===")
print(f"Vision tokens: {len(vision_tokens)}")
print(f"Text tokens: {len(text_tokens)}")
print(f"Total: {len(vision_tokens) + len(text_tokens)}")

# Analyze vision token positions
vision_positions = sorted(vision_tokens['token_position'].unique())
print(f"\n=== VISION TOKEN POSITIONS ===")
print(f"Count: {len(vision_positions)}")
print(f"Min position: {min(vision_positions)}")
print(f"Max position: {max(vision_positions)}")
print(f"Range: {max(vision_positions) - min(vision_positions) + 1}")

# Check if positions are contiguous
expected_positions = list(range(min(vision_positions), max(vision_positions) + 1))
missing_positions = set(expected_positions) - set(vision_positions)
if missing_positions:
    print(f"Missing positions: {sorted(missing_positions)}")
else:
    print(f"Positions are contiguous: YES")

# Calculate possible grid dimensions
num_vision_tokens = len(vision_positions)
print(f"\n=== GRID DIMENSION ANALYSIS ===")
print(f"Number of vision tokens: {num_vision_tokens}")

# Test various grid sizes
sqrt_val = np.sqrt(num_vision_tokens)
print(f"Square root: {sqrt_val:.4f}")

# Find all factors
factors = []
for i in range(1, int(np.sqrt(num_vision_tokens)) + 2):
    if num_vision_tokens % i == 0:
        j = num_vision_tokens // i
        factors.append((i, j))

if sqrt_val == int(sqrt_val):
    print(f"Perfect square: {int(sqrt_val)}x{int(sqrt_val)}")
    grid_dims = (int(sqrt_val), int(sqrt_val))
else:
    print(f"NOT a perfect square")
    
    # Show possible rectangular grids
    print(f"\nPossible rectangular grids:")
    for i, j in factors:
        print(f"  {i}x{j} = {i*j}")
    
    # Find closest to square
    grid_dims = min(factors, key=lambda x: abs(x[0] - x[1]))
    print(f"\nClosest to square: {grid_dims[0]}x{grid_dims[1]}")

# Show first few and last few vision tokens
print(f"\n=== SAMPLE VISION TOKENS ===")
print("First 10 positions:")
for pos in vision_positions[:10]:
    token = vision_tokens[vision_tokens['token_position'] == pos]['token_masked'].iloc[0]
    print(f"  Pos {pos}: {token}")

print("\nLast 10 positions:")
for pos in vision_positions[-10:]:
    token = vision_tokens[vision_tokens['token_position'] == pos]['token_masked'].iloc[0]
    print(f"  Pos {pos}: {token}")

# Check text token positions
text_positions = sorted(text_tokens['token_position'].unique())
print(f"\n=== TEXT TOKEN POSITIONS ===")
print(f"Count: {len(text_positions)}")
if text_positions:
    print(f"Min position: {min(text_positions)}")
    print(f"Max position: {max(text_positions)}")
    print(f"\nFirst 10 text tokens:")
    for pos in text_positions[:10]:
        token = text_tokens[text_tokens['token_position'] == pos]['token_masked'].iloc[0]
        # Show repr to see special characters
        print(f"  Pos {pos}: {repr(token)}")

# Image resolution analysis
print(f"\n=== IMAGE RESOLUTION ANALYSIS ===")
print(f"Image was: 1024x1024 -> downscaled to 768x768")
print(f"Vision tokens found: {num_vision_tokens}")

# Common patch sizes for vision transformers
print(f"\nCommon ViT patch size calculations for 768x768:")
for patch_size in [14, 16, 28, 32]:
    patches_per_side = 768 // patch_size
    total_patches = patches_per_side * patches_per_side
    print(f"  Patch size {patch_size}x{patch_size}: {patches_per_side}x{patches_per_side} = {total_patches} patches")
    if total_patches == num_vision_tokens:
        print(f"    [MATCH!]")

# Qwen3-VL specific: they might use adaptive resolution
print(f"\nIf using non-uniform patches or adaptive resolution:")
print(f"  Actual tokens: {num_vision_tokens}")
print(f"  Expected for 24x24: 576 (you were expecting this)")
print(f"  Difference: {num_vision_tokens - 576}")

print(f"\n=== SUMMARY ===")
print(f"The grid IS: {grid_dims[0]}x{grid_dims[1]}")
print(f"This gives us {num_vision_tokens} tokens")
if num_vision_tokens == 576:
    print(f"  -> This matches your expectation of 24x24 = 576 tokens!")
    print(f"  -> Corresponds to 768x768 image with 32x32 pixel patches")
else:
    print(f"  -> Expected 576 (24x24), got {num_vision_tokens} (difference: {num_vision_tokens - 576})")
print(f"\nThe visualization code will display as: {int(np.ceil(sqrt_val))}x{int(np.ceil(sqrt_val))} grid")

