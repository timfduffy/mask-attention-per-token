import pandas as pd

# Verify the orb_looking_at split file
df = pd.read_parquet('output/eras_text_orb_looking_at_orb_looking_at.parquet')

print(f"Rows: {len(df):,}")
print(f"Unique prompts: {df['prompt_name'].unique()}")

# Check vision tokens for this prompt
vision = df[(df['layer']==0) & (df['generation_step']==0) & (df['variant']=='Full')]
vision_tokens = vision[vision['token_masked'].str.contains('image_pad', case=False)]

print(f"Vision tokens (layer 0, step 0, Full): {len(vision_tokens['token_position'].unique())}")
print(f"Expected: 576 (24x24 grid)")
print(f"Match: {'YES' if len(vision_tokens['token_position'].unique()) == 576 else 'NO'}")

