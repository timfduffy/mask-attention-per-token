"""
Helper script to convert Parquet files to JSON format
Use this if the browser-based Parquet viewer isn't working
"""

import pandas as pd
import sys
import os
from pathlib import Path

def convert_parquet_to_json(parquet_file):
    """Convert a Parquet file to JSON format"""
    
    # Check if file exists
    if not os.path.exists(parquet_file):
        print(f"Error: File not found: {parquet_file}")
        return False
    
    # Generate output filename
    json_file = parquet_file.replace('.parquet', '.json')
    
    print(f"Converting: {parquet_file}")
    print(f"       to: {json_file}")
    
    try:
        # Load Parquet
        print("Loading Parquet file...")
        df = pd.read_parquet(parquet_file)
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Save to JSON
        print("Saving to JSON...")
        df.to_json(json_file, orient='records')
        
        # Show file sizes
        parquet_size = os.path.getsize(parquet_file) / (1024 * 1024)
        json_size = os.path.getsize(json_file) / (1024 * 1024)
        
        print(f"\nSuccess!")
        print(f"  Parquet: {parquet_size:.2f} MB")
        print(f"  JSON:    {json_size:.2f} MB ({json_size/parquet_size:.1f}x larger)")
        print(f"\nYou can now load {json_file} in visualize_results_json.html")
        
        return True
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Parquet to JSON Converter")
        print("=" * 50)
        print("\nUsage:")
        print("  python convert_parquet_to_json.py <file.parquet>")
        print("\nExample:")
        print("  python convert_parquet_to_json.py output/my_results.parquet")
        print("\nThis will create: output/my_results.json")
        print("\nNote: JSON files are typically 200-300x larger than Parquet files,")
        print("      but may be easier to load in some browsers.")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    success = convert_parquet_to_json(parquet_file)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

