"""
Check all parquet files in output folder for multiple prompts
"""
import pandas as pd
from pathlib import Path

output_dir = Path('output')
parquet_files = sorted(output_dir.glob('*.parquet'))

print("=" * 70)
print("CHECKING ALL PARQUET FILES FOR MULTIPLE PROMPTS")
print("=" * 70)

files_to_split = []

for filepath in parquet_files:
    try:
        df = pd.read_parquet(filepath)
        
        # Check if prompt_name column exists
        if 'prompt_name' not in df.columns:
            print(f"\n{filepath.name}: No prompt_name column (single-prompt file)")
            continue
        
        # Get unique prompts
        unique_prompts = df['prompt_name'].unique()
        num_prompts = len(unique_prompts)
        
        if num_prompts > 1:
            print(f"\n{filepath.name}: {num_prompts} prompts - NEEDS SPLITTING")
            for prompt in unique_prompts:
                prompt_count = len(df[df['prompt_name'] == prompt])
                print(f"  - {prompt}: {prompt_count:,} rows")
            files_to_split.append(filepath)
        else:
            print(f"\n{filepath.name}: 1 prompt ({unique_prompts[0]}) - already split")
            
    except Exception as e:
        print(f"\n{filepath.name}: Error reading file - {e}")

print("\n" + "=" * 70)
print(f"SUMMARY: {len(files_to_split)} file(s) need splitting")
print("=" * 70)

if files_to_split:
    print("\nFiles to split:")
    for f in files_to_split:
        print(f"  - {f.name}")
    
    print("\nRun this command to split them:")
    file_list = ' '.join([f'"{f.name}"' for f in files_to_split])
    print(f"  python split_parquet_by_prompt.py {file_list}")
else:
    print("\nAll parquet files are already split or single-prompt!")

