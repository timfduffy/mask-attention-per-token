"""
Deep dive into vision token layout - check for gaps or special positions
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
text_tokens = step_0[~step_0['token_masked'].str.contains('vision_token|image_pad|img_', case=False, na=False)]

print("=== CHECKING FOR GAPS IN VISION TOKEN POSITIONS ===\n")

# Get all vision token positions
vision_positions = sorted(vision_tokens['token_position'].unique())
print(f"Vision token positions: {vision_positions[0]} to {vision_positions[-1]}")
print(f"Total vision positions: {len(vision_positions)}")

# Check each position in the range
min_pos = vision_positions[0]
max_pos = vision_positions[-1]

print(f"\nScanning positions {min_pos} to {max_pos} for gaps or non-vision tokens:")
non_vision_in_range = []

for pos in range(min_pos, max_pos + 1):
    # Check what token is at this position
    token_at_pos = step_0[step_0['token_position'] == pos]
    
    if len(token_at_pos) == 0:
        print(f"  Position {pos}: NO DATA (missing entirely)")
        non_vision_in_range.append(pos)
    else:
        token_str = token_at_pos['token_masked'].iloc[0]
        is_vision = 'vision_token' in token_str.lower() or 'image_pad' in token_str.lower() or token_str.startswith('img_')
        
        if not is_vision:
            print(f"  Position {pos}: TEXT TOKEN '{token_str}' (embedded in vision range!)")
            non_vision_in_range.append(pos)

if non_vision_in_range:
    print(f"\nFound {len(non_vision_in_range)} non-vision positions in vision range: {non_vision_in_range}")
    print(f"\nActual continuous vision token positions:")
    
    # Find contiguous ranges
    ranges = []
    start = vision_positions[0]
    prev = start
    
    for pos in vision_positions[1:]:
        if pos != prev + 1:
            # Gap detected
            ranges.append((start, prev))
            start = pos
        prev = pos
    ranges.append((start, prev))
    
    for i, (start, end) in enumerate(ranges, 1):
        count = end - start + 1
        print(f"  Range {i}: positions {start} to {end} ({count} tokens)")
    
    print(f"\nTotal vision tokens across all ranges: {sum(end - start + 1 for start, end in ranges)}")
else:
    print(f"\nAll positions {min_pos} to {max_pos} are vision tokens (no gaps)")

# Show token distribution
print(f"\n=== OVERALL TOKEN DISTRIBUTION ===")
all_positions = sorted(step_0['token_position'].unique())
print(f"All token positions: {all_positions[0]} to {all_positions[-1]}")

print(f"\nToken type by position range:")
for pos in all_positions:
    token_str = step_0[step_0['token_position'] == pos]['token_masked'].iloc[0]
    is_vision = 'vision_token' in token_str.lower() or 'image_pad' in token_str.lower() or token_str.startswith('img_')
    token_type = "VISION" if is_vision else "TEXT  "
    
    # Only show first 5, last 5, and around special positions
    if pos < 10 or pos > all_positions[-5] or pos in range(530, 540) or pos in range(547, 552):
        print(f"  Pos {pos:3d}: {token_type} - '{token_str}'")
    elif pos == 10:
        print(f"  ... ({len([p for p in all_positions if 10 < p < 530])} more positions)")

