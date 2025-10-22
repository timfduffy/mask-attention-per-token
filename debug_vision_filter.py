"""
Debug the vision token filter to see what it's actually matching
"""
import pandas as pd

df = pd.read_parquet('output/eras_text_orb_looking_at.parquet')
layer0 = df[(df['layer']==0) & (df['generation_step']==0)]

print("Testing vision token filters...\n")

# Original filter
original_filter = layer0['token_masked'].str.contains('vision_token|image_pad|img_', case=False, na=False)
original_matches = layer0[original_filter]

print(f"Original filter matches: {len(original_matches)} tokens")
print(f"Unique tokens matched:")
for token in sorted(original_matches['token_masked'].unique()):
    count = len(original_matches[original_matches['token_masked'] == token])
    print(f"  '{token}' ({count} times)")

# Check specific positions
print(f"\nChecking specific positions:")
for pos in [3, 4, 531, 532, 549, 550, 579, 580]:
    token_data = layer0[layer0['token_position'] == pos]
    if len(token_data) > 0:
        token_str = token_data['token_masked'].iloc[0]
        matched = original_filter.iloc[token_data.index[0]]
        print(f"  Pos {pos}: '{token_str}' - Matched: {matched}")

# Better filter
print(f"\n\nBetter filter (excluding start/end markers):")
better_filter = (
    layer0['token_masked'].str.contains('image_pad', case=False, na=False) |
    layer0['token_masked'].str.contains('img_', case=False, na=False)
) & ~layer0['token_masked'].str.contains('vision_start|vision_end', case=False, na=False)

better_matches = layer0[better_filter]
print(f"Better filter matches: {len(better_matches)} tokens")
print(f"Unique tokens matched:")
for token in sorted(better_matches['token_masked'].unique()):
    count = len(better_matches[better_matches['token_masked'] == token])
    print(f"  '{token}' ({count} times)")

