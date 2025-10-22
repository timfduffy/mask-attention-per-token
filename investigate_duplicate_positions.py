"""
Investigate why positions have duplicate entries
"""
import pandas as pd

df = pd.read_parquet('output/eras_text_orb_looking_at.parquet')

# Get single layer/step/variant
subset = df[(df['layer']==0) & (df['generation_step']==0) & (df['variant']=='Full')]

print("=== INVESTIGATING DUPLICATE POSITIONS ===\n")

# Check position 549 in detail
pos_549 = subset[subset['token_position']==549]
print(f"Position 549 appears {len(pos_549)} times:\n")
print(pos_549[['token_position', 'token_masked', 'prompt_name', 'l2_distance']].to_string())

# Check unique prompt names
print(f"\n\nUnique prompt names: {subset['prompt_name'].unique()}")
print(f"Number of prompts: {len(subset['prompt_name'].unique())}")

# Count rows per prompt
print(f"\nRows per prompt:")
for prompt in subset['prompt_name'].unique():
    prompt_rows = subset[subset['prompt_name']==prompt]
    print(f"  {prompt}: {len(prompt_rows)} rows")

# Check if different prompts have different sequences
print(f"\n\n=== CHECKING EACH PROMPT SEPARATELY ===\n")
for prompt in subset['prompt_name'].unique():
    prompt_data = subset[subset['prompt_name']==prompt]
    
    vision = prompt_data[prompt_data['token_masked'].str.contains('image_pad', case=False, na=False)]
    text = prompt_data[~prompt_data['token_masked'].str.contains('image_pad', case=False, na=False)]
    
    print(f"\nPrompt: {prompt}")
    print(f"  Total rows: {len(prompt_data)}")
    print(f"  Vision tokens: {len(vision)}")
    print(f"  Text tokens: {len(text)}")
    print(f"  Unique positions: {len(prompt_data['token_position'].unique())}")
    
    # Check position 549 for this prompt
    pos_549_here = prompt_data[prompt_data['token_position']==549]
    if len(pos_549_here) > 0:
        token = pos_549_here['token_masked'].iloc[0]
        is_vision = 'image_pad' in token.lower()
        print(f"  Position 549: {('VISION' if is_vision else 'TEXT')} - {repr(token)}")

