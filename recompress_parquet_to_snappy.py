"""
Convert existing ZSTD-compressed Parquet files to Snappy compression
for browser compatibility with hyparquet
"""

import pandas as pd
import sys
import os
from pathlib import Path

def recompress_to_snappy(input_file):
    """Re-compress a Parquet file from ZSTD to Snappy"""
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return False
    
    # Check current compression
    import pyarrow.parquet as pq
    try:
        meta = pq.read_metadata(input_file)
        current_compression = meta.row_group(0).column(0).compression
        print(f"Current compression: {current_compression}")
        
        if current_compression == 'SNAPPY':
            print("File already uses Snappy compression - no conversion needed!")
            return True
    except Exception as e:
        print(f"Warning: Could not read metadata: {e}")
    
    # Create backup
    backup_file = input_file + '.zstd.backup'
    print(f"\nCreating backup: {backup_file}")
    
    try:
        # Load the file
        print(f"Loading: {input_file}")
        df = pd.read_parquet(input_file)
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Backup original (rename)
        os.rename(input_file, backup_file)
        
        # Save with Snappy compression
        print(f"\nSaving with Snappy compression: {input_file}")
        df.to_parquet(input_file, compression='snappy', index=False)
        
        # Check sizes
        old_size = os.path.getsize(backup_file) / (1024 * 1024)
        new_size = os.path.getsize(input_file) / (1024 * 1024)
        
        print(f"\nSuccess!")
        print(f"  ZSTD size:   {old_size:.2f} MB")
        print(f"  Snappy size: {new_size:.2f} MB ({new_size/old_size*100:.1f}%)")
        print(f"\nBackup saved as: {backup_file}")
        print(f"You can delete the backup once you verify the new file works!")
        
        return True
        
    except Exception as e:
        print(f"\nError during conversion: {e}")
        # Restore backup if it exists
        if os.path.exists(backup_file):
            print("Restoring backup...")
            if os.path.exists(input_file):
                os.remove(input_file)
            os.rename(backup_file, input_file)
            print("Backup restored.")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Re-compress Parquet from ZSTD to Snappy")
        print("=" * 50)
        print("\nUsage:")
        print("  python recompress_parquet_to_snappy.py <file.parquet>")
        print("\nExample:")
        print("  python recompress_parquet_to_snappy.py output/image_question.parquet")
        print("\nThis will:")
        print("  1. Backup the original file (adds .zstd.backup extension)")
        print("  2. Re-save with Snappy compression")
        print("  3. Make the file browser-compatible with hyparquet")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    success = recompress_to_snappy(parquet_file)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()

