"""
Quick script to convert existing image_question.csv to Parquet
"""

import pandas as pd
import os
from pathlib import Path

print("Converting image_question.csv to Parquet...")

csv_file = Path('output/image_question.csv')
parquet_file = Path('output/image_question.parquet')

if not csv_file.exists():
    print(f"Error: {csv_file} not found")
    exit(1)

print(f"\nLoading CSV file...")
df = pd.read_csv(csv_file)

print(f"Loaded {len(df):,} rows")
print(f"CSV size: {csv_file.stat().st_size / (1024**2):.2f} MB")

# Optimize like the main code does
print(f"\nOptimizing data...")
df_optimized = df.copy()

# Convert float columns to float32 (round to 4 decimal places)
for col in ['l2_distance', 'cosine_distance']:
    if col in df_optimized.columns:
        df_optimized[col] = df_optimized[col].round(4).astype('float32')
        print(f"  Converted {col} to float32 (4 decimal precision)")

# Convert string columns to categorical for better compression
string_cols = df_optimized.select_dtypes(include=['object']).columns.tolist()
for col in string_cols:
    unique_ratio = df_optimized[col].nunique() / len(df_optimized)
    if unique_ratio < 0.5:  # Only if less than 50% unique values
        df_optimized[col] = df_optimized[col].astype('category')
        print(f"  Converted {col} to categorical ({df_optimized[col].nunique()} unique values)")

print(f"\nSaving Parquet...")
df_optimized.to_parquet(parquet_file, compression='zstd', index=False)

parquet_size = parquet_file.stat().st_size
csv_size = csv_file.stat().st_size

print(f"\nDone!")
print(f"  CSV:     {csv_size / (1024**2):.2f} MB")
print(f"  Parquet: {parquet_size / (1024**2):.2f} MB")
print(f"  Reduction: {(1 - parquet_size/csv_size)*100:.1f}%")
print(f"\nFile saved to: {parquet_file}")
print(f"\nNow open visualize_results.html and load this Parquet file!")

