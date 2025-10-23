"""
Fix cosine_distance precision in existing Parquet files
Increases precision from 4 to 6 decimal places to avoid losing small values
"""
import pandas as pd
import sys
from pathlib import Path

def fix_cosine_precision(input_file):
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)
    
    print(f"Loading: {input_file}")
    df = pd.read_parquet(input_path)
    print(f"  Rows: {len(df):,}")
    
    if 'cosine_distance' not in df.columns:
        print("Error: No cosine_distance column found!")
        sys.exit(1)
    
    # Show current stats
    print(f"\nBEFORE fix (4 decimal precision):")
    print(f"  Zeros: {(df['cosine_distance'] == 0).sum():,} ({(df['cosine_distance'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Min: {df['cosine_distance'].min():.6f}")
    print(f"  Median: {df['cosine_distance'].median():.6f}")
    print(f"  75th percentile: {df['cosine_distance'].quantile(0.75):.6f}")
    print(f"  Max: {df['cosine_distance'].max():.6f}")
    
    # Create backup
    backup_file = input_path.parent / f"{input_path.stem}.4decimals.backup"
    print(f"\nCreating backup: {backup_file}")
    input_path.rename(backup_file)
    
    # Reload from backup with full precision
    print(f"Reprocessing with 6 decimal precision...")
    df_full = pd.read_parquet(backup_file)
    
    # Re-optimize with correct precision
    df_optimized = df_full.copy()
    
    # L2 distance: keep 4 decimals
    if 'l2_distance' in df_optimized.columns:
        df_optimized['l2_distance'] = df_optimized['l2_distance'].round(4).astype('float32')
    
    # Cosine distance: increase to 6 decimals
    if 'cosine_distance' in df_optimized.columns:
        df_optimized['cosine_distance'] = df_optimized['cosine_distance'].round(6).astype('float32')
    
    # Keep categorical encoding
    string_cols = df_optimized.select_dtypes(include=['object']).columns.tolist()
    for col in string_cols:
        df_optimized[col] = df_optimized[col].astype('category')
    
    # Save with Snappy compression
    print(f"\nSaving fixed file: {input_file}")
    df_optimized.to_parquet(input_file, compression='snappy', index=False)
    
    # Show new stats
    df_new = pd.read_parquet(input_file)
    print(f"\nAFTER fix (6 decimal precision):")
    print(f"  Zeros: {(df_new['cosine_distance'] == 0).sum():,} ({(df_new['cosine_distance'] == 0).sum() / len(df_new) * 100:.1f}%)")
    print(f"  Min: {df_new['cosine_distance'].min():.6f}")
    print(f"  Median: {df_new['cosine_distance'].median():.6f}")
    print(f"  75th percentile: {df_new['cosine_distance'].quantile(0.75):.6f}")
    print(f"  Max: {df_new['cosine_distance'].max():.6f}")
    
    # Compare file sizes
    print(f"\nFile sizes:")
    print(f"  Before: {backup_file.stat().st_size / (1024*1024):.2f} MB")
    print(f"  After:  {input_path.stat().st_size / (1024*1024):.2f} MB ({input_path.stat().st_size / backup_file.stat().st_size * 100:.1f}%)")
    
    print(f"\nSuccess! Backup saved as: {backup_file}")
    print("You can delete the backup once you verify the new file works!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Fix Cosine Distance Precision")
        print("=" * 50)
        print("\nUsage:")
        print("  python fix_cosine_precision.py <file.parquet>")
        print("\nExample:")
        print("  python fix_cosine_precision.py output/dogcat_dogcatsmall.parquet")
        print("\nThis script:")
        print("  - Creates a backup with .4decimals.backup extension")
        print("  - Increases cosine_distance precision from 4 to 6 decimals")
        print("  - Keeps l2_distance at 4 decimals")
        print("  - Maintains categorical encoding and Snappy compression")
        sys.exit(1)
    
    fix_cosine_precision(sys.argv[1])

