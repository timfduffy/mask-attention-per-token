"""
Create a smaller sample Parquet file for testing browser viewing
"""

import pandas as pd
import sys

def create_sample(input_file, output_file, max_rows=10000):
    """Create a smaller sample of a Parquet file"""
    
    print(f"Loading: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"  Original: {len(df):,} rows")
    
    # Take first N rows
    df_sample = df.head(max_rows)
    print(f"  Sample: {len(df_sample):,} rows")
    
    # Save
    print(f"\nSaving: {output_file}")
    df_sample.to_parquet(output_file, compression='snappy', index=False)
    
    import os
    size = os.path.getsize(output_file) / 1024
    print(f"  Size: {size:.1f} KB")
    print("\nDone! Try loading this file in the browser.")

if __name__ == '__main__':
    input_file = 'output/image_question.parquet'
    output_file = 'output/image_question_sample.parquet'
    max_rows = 10000
    
    if len(sys.argv) > 1:
        max_rows = int(sys.argv[1])
    
    create_sample(input_file, output_file, max_rows)

