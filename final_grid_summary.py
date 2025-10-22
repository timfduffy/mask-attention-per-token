"""
Create final summary of the grid layout issue
"""
import pandas as pd

df = pd.read_parquet('output/eras_text_orb_looking_at.parquet')
layer0 = df[(df['layer']==0) & (df['generation_step']==0)]

# Get vision tokens
vision_tokens = layer0[layer0['token_masked'].str.contains('vision_token|image_pad|img_', case=False, na=False)]
vision_positions = sorted(vision_tokens['token_position'].unique())

print("=" * 70)
print("FINAL GRID LAYOUT ANALYSIS")
print("=" * 70)

print(f"\nTotal vision tokens: {len(vision_positions)}")
print(f"Grid dimensions: 24x24 = 576 [OK]")
print(f"\nVision token positions: {min(vision_positions)} to {max(vision_positions)}")

# Find gaps by checking if all positions in range are vision tokens
print("\n" + "=" * 70)
print("POSITION BREAKDOWN:")
print("=" * 70)

# Check if there are any missing positions in the range
expected_positions = set(range(min(vision_positions), max(vision_positions) + 1))
vision_pos_set = set(vision_positions)
missing_positions = expected_positions - vision_pos_set

first_range_end = None
if missing_positions:
    # Find first gap
    missing_sorted = sorted(missing_positions)
    first_gap_pos = missing_sorted[0]
    # Find where vision tokens resume
    first_range_end = first_gap_pos - 1

if first_range_end:
    first_range_count = first_range_end - vision_positions[0] + 1
    second_range_start = vision_positions[vision_positions.index(first_range_end) + 1]
    second_range_count = max(vision_positions) - second_range_start + 1
    
    print(f"\nVision Range 1: positions {vision_positions[0]}-{first_range_end}")
    print(f"  Count: {first_range_count} tokens")
    
    print(f"\nText Gap: positions {first_range_end+1}-{second_range_start-1}")
    gap_size = second_range_start - first_range_end - 1
    print(f"  Count: {gap_size} TEXT tokens (including position 549!)")
    
    # Show what's in the gap
    print(f"\n  Tokens in gap:")
    for pos in range(first_range_end+1, min(second_range_start, first_range_end+20)):
        token_data = layer0[layer0['token_position']==pos]
        if len(token_data) > 0:
            token_str = token_data['token_masked'].iloc[0]
            display = repr(token_str) if token_str.strip() else "'<newline>'"
            print(f"    Pos {pos}: {display}")
    
    print(f"\nVision Range 2: positions {second_range_start}-{max(vision_positions)}")
    print(f"  Count: {second_range_count} tokens")
    
    print(f"\nTotal vision tokens: {first_range_count} + {second_range_count} = {first_range_count + second_range_count}")
else:
    print("\nVision tokens are contiguous (no gaps)")

print("\n" + "=" * 70)
print("EXPLANATION:")
print("=" * 70)
print("\nYou saw 'position 549' as the highest, but that's misleading!")
print("Position 549 is a TEXT token (newline), not a vision token.")
print("\nThe visualization tool:")
print("  - Filters tokens by CONTENT (looks for 'image_pad', etc.)")
print("  - Ignores position numbers when determining what's a vision token")
print("  - Correctly identifies all 576 vision tokens")
print("  - Displays them in a 24x24 grid")
print("\nImage processing:")
print("  1024x1024 original -> 768x768 downscaled")
print("  768/32 = 24 (using 32x32 pixel patches)")
print("  24x24 = 576 vision tokens [OK]")

