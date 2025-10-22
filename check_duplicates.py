"""
Check if there are duplicate positions in the data
"""
import pandas as pd

df = pd.read_parquet('output/eras_text_orb_looking_at.parquet')

# Get single layer/step/variant
subset = df[(df['layer']==0) & (df['generation_step']==0) & (df['variant']=='Full')]

print(f"Total rows: {len(subset)}")
print(f"Unique positions: {len(subset['token_position'].unique())}")

# Check a specific position
pos_549_rows = subset[subset['token_position']==549]
print(f"\nRows with position 549: {len(pos_549_rows)}")
if len(pos_549_rows) > 0:
    print(pos_549_rows[['token_position', 'token_masked', 'l2_distance']].to_string())

# Count rows per position
pos_counts = subset['token_position'].value_counts()
print(f"\nPosition value counts:")
print(f"  Min count: {pos_counts.min()}")
print(f"  Max count: {pos_counts.max()}")
print(f"  Mode: {pos_counts.mode().iloc[0]}")

# Show positions with multiple rows
multi_rows = pos_counts[pos_counts > 1]
if len(multi_rows) > 0:
    print(f"\nPositions with multiple rows: {len(multi_rows)}")
    print(f"Examples:")
    for pos in list(multi_rows.index[:5]):
        print(f"  Position {pos}: {pos_counts[pos]} rows")
else:
    print(f"\nAll positions appear exactly once - good!")

# Check: are there really multiple generation steps?
print(f"\nGeneration steps in subset: {sorted(subset['generation_step'].unique())}")
print(f"Variants in subset: {sorted(subset['variant'].unique())}")
print(f"Layers in subset: {sorted(subset['layer'].unique())}")

