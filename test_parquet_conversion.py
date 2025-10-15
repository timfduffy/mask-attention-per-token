"""
Test script to convert CSV to optimized Parquet format
Compares file sizes and loading speeds
"""

import pandas as pd
import time
import os
from pathlib import Path

def format_size(bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def test_conversion(csv_file):
    """Test different Parquet configurations"""
    
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Memory usage: {format_size(df.memory_usage(deep=True).sum())}")
    
    # Get original CSV size
    csv_size = os.path.getsize(csv_file)
    print(f"\nOriginal CSV file size: {format_size(csv_size)}")
    
    # Identify numeric and string columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumeric columns: {numeric_cols}")
    print(f"String columns: {string_cols}")
    
    # Test 1: Original JSON format (for comparison)
    print("\n" + "="*70)
    print("Test 1: JSON (current format)")
    print("="*70)
    json_file = 'test_output.json'
    start = time.time()
    df.to_json(json_file, orient='records')
    json_time = time.time() - start
    json_size = os.path.getsize(json_file)
    print(f"Size: {format_size(json_size)} ({json_size/csv_size*100:.1f}% of CSV)")
    print(f"Write time: {json_time:.3f}s")
    
    # Test 2: Basic Parquet (no optimizations)
    print("\n" + "="*70)
    print("Test 2: Basic Parquet (float64, no compression)")
    print("="*70)
    parquet_basic = 'test_basic.parquet'
    start = time.time()
    df.to_parquet(parquet_basic, compression=None, index=False)
    basic_time = time.time() - start
    basic_size = os.path.getsize(parquet_basic)
    print(f"Size: {format_size(basic_size)} ({basic_size/csv_size*100:.1f}% of CSV)")
    print(f"Write time: {basic_time:.3f}s")
    
    # Test 3: Parquet with Snappy compression
    print("\n" + "="*70)
    print("Test 3: Parquet with Snappy compression (float64)")
    print("="*70)
    parquet_snappy = 'test_snappy.parquet'
    start = time.time()
    df.to_parquet(parquet_snappy, compression='snappy', index=False)
    snappy_time = time.time() - start
    snappy_size = os.path.getsize(parquet_snappy)
    print(f"Size: {format_size(snappy_size)} ({snappy_size/csv_size*100:.1f}% of CSV)")
    print(f"Write time: {snappy_time:.3f}s")
    print(f"Reduction from JSON: {(1 - snappy_size/json_size)*100:.1f}%")
    
    # Test 4: Optimized Parquet (float32 + categorical + snappy)
    print("\n" + "="*70)
    print("Test 4: Optimized Parquet (float32 + categorical + snappy)")
    print("="*70)
    df_optimized = df.copy()
    
    # Convert float columns to float32 (round to 4 decimal places)
    for col in numeric_cols:
        if 'distance' in col.lower() or 'float' in str(df[col].dtype):
            df_optimized[col] = df_optimized[col].round(4).astype('float32')
            print(f"  Converted {col} to float32 (rounded to 4 decimals)")
    
    # Convert string columns to categorical
    for col in string_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5:  # Only if less than 50% unique values
            df_optimized[col] = df_optimized[col].astype('category')
            print(f"  Converted {col} to categorical ({df[col].nunique()} unique values)")
    
    parquet_optimized = 'test_optimized.parquet'
    start = time.time()
    df_optimized.to_parquet(parquet_optimized, compression='snappy', index=False)
    opt_time = time.time() - start
    opt_size = os.path.getsize(parquet_optimized)
    print(f"Size: {format_size(opt_size)} ({opt_size/csv_size*100:.1f}% of CSV)")
    print(f"Write time: {opt_time:.3f}s")
    print(f"Reduction from JSON: {(1 - opt_size/json_size)*100:.1f}%")
    print(f"Reduction from basic Parquet: {(1 - opt_size/basic_size)*100:.1f}%")
    
    # Test 5: Optimized Parquet with ZSTD (best compression, but not browser-compatible)
    print("\n" + "="*70)
    print("Test 5: Optimized Parquet (float32 + categorical + ZSTD)")
    print("Note: ZSTD has best compression but requires hyparquet-compressors in browser")
    print("="*70)
    parquet_zstd = 'test_zstd.parquet'
    start = time.time()
    df_optimized.to_parquet(parquet_zstd, compression='zstd', index=False)
    zstd_time = time.time() - start
    zstd_size = os.path.getsize(parquet_zstd)
    print(f"Size: {format_size(zstd_size)} ({zstd_size/csv_size*100:.1f}% of CSV)")
    print(f"Write time: {zstd_time:.3f}s")
    print(f"Reduction from JSON: {(1 - zstd_size/json_size)*100:.1f}%")
    
    # Test loading speeds
    print("\n" + "="*70)
    print("Loading Speed Comparison")
    print("="*70)
    
    # JSON loading
    start = time.time()
    df_json = pd.read_json(json_file)
    json_load_time = time.time() - start
    print(f"JSON load time: {json_load_time:.3f}s")
    
    # Parquet optimized loading
    start = time.time()
    df_parquet = pd.read_parquet(parquet_optimized)
    parquet_load_time = time.time() - start
    print(f"Parquet (optimized) load time: {parquet_load_time:.3f}s")
    print(f"Speedup: {json_load_time/parquet_load_time:.2f}x faster")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original CSV:        {format_size(csv_size)}")
    print(f"JSON (current):      {format_size(json_size)} ({json_size/csv_size*100:.1f}%)")
    print(f"Parquet (optimized): {format_size(opt_size)} ({opt_size/csv_size*100:.1f}%)")
    print(f"Parquet (ZSTD):      {format_size(zstd_size)} ({zstd_size/csv_size*100:.1f}%)")
    print(f"\nBest size reduction: {(1 - zstd_size/json_size)*100:.1f}% smaller than JSON")
    print(f"Best speed improvement: {json_load_time/parquet_load_time:.2f}x faster loading")
    
    # Verify data integrity
    print("\n" + "="*70)
    print("Data Integrity Check")
    print("="*70)
    df_check = pd.read_parquet(parquet_optimized)
    
    # Check shape
    print(f"Shape match: {df.shape == df_check.shape}")
    
    # Check numeric precision (float32 vs float64)
    for col in numeric_cols:
        if 'distance' in col.lower():
            max_diff = (df[col] - df_check[col]).abs().max()
            print(f"{col} max difference: {max_diff:.6e} (acceptable for 4 decimal places)")
    
    # Cleanup
    print("\nCleaning up test files...")
    for f in [json_file, parquet_basic, parquet_snappy, parquet_optimized, parquet_zstd]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Removed {f}")
    
    print("\n✓ Test complete!")
    
    return {
        'csv_size': csv_size,
        'json_size': json_size,
        'parquet_size': opt_size,
        'parquet_zstd_size': zstd_size,
        'json_load_time': json_load_time,
        'parquet_load_time': parquet_load_time
    }


if __name__ == '__main__':
    import sys
    
    # Look for CSV file
    csv_file = 'output/image_question.csv'
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        print("Please run the VL masking experiment first to generate the CSV file")
        sys.exit(1)
    
    results = test_conversion(csv_file)

