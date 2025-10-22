"""
Split concatenated parquet files into separate files per prompt.

This script reads parquet files that contain multiple prompts mixed together
and splits them into separate files, one per prompt.

Usage:
    python split_parquet_by_prompt.py input_file.parquet
    python split_parquet_by_prompt.py output/*.parquet  # Process multiple files
"""

import pandas as pd
import argparse
from pathlib import Path
import sys


def split_parquet_file(input_file: Path, output_dir: Path = None, dry_run: bool = False, 
                       keep_original: bool = False, delete_original: bool = False):
    """
    Split a parquet file by prompt_name into separate files.
    
    Args:
        input_file: Path to input parquet file
        output_dir: Directory to save split files (default: same as input)
        dry_run: If True, just print what would be done without saving
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*70}")
    
    # Load the parquet file
    try:
        df = pd.read_parquet(input_file)
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return False
    
    print(f"Total rows: {len(df):,}")
    
    # Check if prompt_name column exists
    if 'prompt_name' not in df.columns:
        print(f"[ERROR] No 'prompt_name' column found. This file may already be split or use a different format.")
        print(f"        Columns: {df.columns.tolist()}")
        return False
    
    # Get unique prompts
    unique_prompts = df['prompt_name'].unique()
    print(f"\nFound {len(unique_prompts)} unique prompt(s):")
    
    # Show breakdown
    for prompt_name in unique_prompts:
        prompt_df = df[df['prompt_name'] == prompt_name]
        print(f"  - {prompt_name}: {len(prompt_df):,} rows")
    
    # If only one prompt, no need to split
    if len(unique_prompts) == 1:
        print(f"\n[OK] File contains only one prompt - no splitting needed")
        return True
    
    # Determine output directory
    if output_dir is None:
        output_dir = input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base name (remove extension and any existing prompt suffix)
    base_name = input_file.stem
    
    # Try to detect if the filename already has a pattern like "base_prompt"
    # and extract just the base
    parts = base_name.split('_')
    if len(parts) > 1:
        # Check if last part matches any prompt name
        if parts[-1] in unique_prompts:
            # Remove the last part to get base name
            base_name = '_'.join(parts[:-1])
    
    print(f"\nSplitting into separate files...")
    print(f"Base name: {base_name}")
    print(f"Output directory: {output_dir}")
    
    # Split and save each prompt
    saved_files = []
    for prompt_name in unique_prompts:
        # Filter data for this prompt
        prompt_df = df[df['prompt_name'] == prompt_name]
        
        # Generate output filename
        output_csv = output_dir / f"{base_name}_{prompt_name}.csv"
        output_parquet = output_dir / f"{base_name}_{prompt_name}.parquet"
        
        if dry_run:
            print(f"\n[DRY RUN] Would save {prompt_name}:")
            print(f"  - {output_csv} ({len(prompt_df):,} rows)")
            print(f"  - {output_parquet}")
            continue
        
        # Optimize dataframe
        df_optimized = prompt_df.copy()
        
        # Convert float columns to float32
        for col in ['l2_distance', 'cosine_distance']:
            if col in df_optimized.columns:
                df_optimized[col] = df_optimized[col].round(4).astype('float32')
        
        # Convert string columns to categorical for compression
        string_cols = df_optimized.select_dtypes(include=['object']).columns.tolist()
        for col in string_cols:
            unique_ratio = df_optimized[col].nunique() / len(df_optimized)
            if unique_ratio < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        # Save files
        print(f"\n  Saving {prompt_name}...")
        prompt_df.to_csv(output_csv, index=False)
        df_optimized.to_parquet(output_parquet, compression='snappy', index=False)
        
        csv_size = output_csv.stat().st_size / (1024 * 1024)
        parquet_size = output_parquet.stat().st_size / (1024 * 1024)
        
        print(f"    [OK] {output_csv.name} ({csv_size:.2f} MB)")
        print(f"    [OK] {output_parquet.name} ({parquet_size:.2f} MB)")
        
        saved_files.append((output_csv, output_parquet))
    
    if not dry_run:
        print(f"\n[OK] Successfully split into {len(unique_prompts)} files")
        
        # Handle original file deletion based on flags
        print(f"\nOriginal file: {input_file}")
        if delete_original:
            input_file.unlink()
            print(f"  [OK] Deleted {input_file.name}")
        elif keep_original:
            print(f"  -> Kept original file")
        else:
            # Ask user
            response = input("Delete original concatenated file? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                input_file.unlink()
                print(f"  [OK] Deleted {input_file.name}")
            else:
                print(f"  -> Kept original file")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Split concatenated parquet files into separate files per prompt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split a single file
  python split_parquet_by_prompt.py output/results.parquet
  
  # Split multiple files
  python split_parquet_by_prompt.py output/*.parquet
  
  # Dry run (show what would be done)
  python split_parquet_by_prompt.py --dry-run output/*.parquet
  
  # Specify output directory
  python split_parquet_by_prompt.py --output-dir output/split results.parquet
        """
    )
    
    parser.add_argument('files', nargs='+', help='Parquet file(s) to split')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory (default: same as input)')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Show what would be done without saving')
    parser.add_argument('--keep-original', '-k', action='store_true', help='Keep original files without asking')
    parser.add_argument('--delete-original', '-d', action='store_true', help='Delete original files without asking')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_files = [Path(f) for f in args.files]
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Validate input files
    valid_files = []
    for f in input_files:
        if not f.exists():
            print(f"[WARNING] File not found: {f}")
        elif not f.suffix.lower() == '.parquet':
            print(f"[WARNING] Not a parquet file: {f}")
        else:
            valid_files.append(f)
    
    if not valid_files:
        print("\n[ERROR] No valid parquet files to process")
        sys.exit(1)
    
    print(f"\nProcessing {len(valid_files)} file(s)...")
    if args.dry_run:
        print("(DRY RUN MODE - no files will be modified)")
    
    # Process each file
    success_count = 0
    for input_file in valid_files:
        if split_parquet_file(input_file, output_dir, args.dry_run, 
                             args.keep_original, args.delete_original):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {success_count}/{len(valid_files)} file(s) processed successfully")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

