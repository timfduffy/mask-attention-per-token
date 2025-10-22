"""
Simple check: count vision tokens per position for layer 0, step 0, variant Full
"""
import pandas as pd

df = pd.read_parquet('output/eras_text_orb_looking_at.parquet')

# Get just one layer, one step, one variant
subset = df[(df['layer']==0) & (df['generation_step']==0) & (df['variant']=='Full')]

print(f"Total rows for layer=0, step=0, variant=Full: {len(subset)}")
print(f"\nUnique positions: {len(subset['token_position'].unique())}")
print(f"Position range: {subset['token_position'].min()} to {subset['token_position'].max()}")

# Count vision vs text
vision = subset[subset['token_masked'].str.contains('image_pad', case=False, na=False)]
text = subset[~subset['token_masked'].str.contains('image_pad', case=False, na=False)]

print(f"\nVision tokens: {len(vision)}")
print(f"Text tokens: {len(text)}")

# Check positions around 549
print(f"\nPositions 545-555:")
for pos in range(545, 556):
    row = subset[subset['token_position']==pos]
    if len(row) > 0:
        token = row['token_masked'].iloc[0]
        is_vision = 'image_pad' in token.lower()
        marker = "VISION" if is_vision else "TEXT  "
        print(f"  Pos {pos}: {marker} - {repr(token)}")

# Get vision token positions
vision_pos = sorted(vision['token_position'].unique())
print(f"\nVision token positions:")
print(f"  Count: {len(vision_pos)}")
print(f"  First 10: {vision_pos[:10]}")
print(f"  Last 10: {vision_pos[-10:]}")
print(f"  Sqrt: {len(vision_pos)**0.5}")

# Check for gaps
all_pos_in_range = set(range(min(vision_pos), max(vision_pos)+1))
missing = all_pos_in_range - set(vision_pos)
if missing:
    print(f"\nMissing positions (gaps): {sorted(missing)}")
else:
    print(f"\nNo gaps - vision tokens are contiguous")

