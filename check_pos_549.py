import pandas as pd

df = pd.read_parquet('output/eras_text_orb_looking_at.parquet')
layer0 = df[(df['layer']==0) & (df['generation_step']==0)]

# Check position 549
token549 = layer0[layer0['token_position']==549]
if len(token549) > 0:
    token_str = token549['token_masked'].iloc[0]
    print(f"Token at position 549: {repr(token_str)}")
    print(f"Is image_pad: {'image_pad' in token_str.lower()}")
    print(f"Is vision_token: {'vision_token' in token_str.lower()}")
else:
    print("No token found at position 549")

# Check positions around 549
print("\nTokens around position 549:")
for pos in range(545, 555):
    token_data = layer0[layer0['token_position']==pos]
    if len(token_data) > 0:
        token_str = token_data['token_masked'].iloc[0]
        is_vision = 'image_pad' in token_str.lower() or 'vision_token' in token_str.lower() or token_str.startswith('img_')
        marker = "VISION" if is_vision else "TEXT  "
        print(f"  Pos {pos}: {marker} - {repr(token_str)}")

