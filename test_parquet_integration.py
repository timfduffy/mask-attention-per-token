"""
Quick test to verify Parquet integration works
"""

import pandas as pd
import os
from pathlib import Path

def test_parquet_roundtrip():
    """Test that we can save and load Parquet files correctly"""
    
    print("Testing Parquet integration...")
    
    # Create sample data similar to actual results
    sample_data = {
        'generation_step': [0, 0, 1, 1] * 10,
        'layer': list(range(4)) * 10,
        'token_masked': ['hello', 'world', 'test', 'token'] * 10,
        'token_position': list(range(4)) * 10,
        'variant': ['Full', 'Attn'] * 20,
        'l2_distance': [0.1234567, 0.2345678, 0.3456789, 0.4567890] * 10,
        'cosine_distance': [0.9876543, 0.8765432, 0.7654321, 0.6543210] * 10
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"Original data: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Optimize like the main code does
    df_optimized = df.copy()
    df_optimized['l2_distance'] = df_optimized['l2_distance'].round(4).astype('float32')
    df_optimized['cosine_distance'] = df_optimized['cosine_distance'].round(4).astype('float32')
    
    for col in ['token_masked', 'variant']:
        df_optimized[col] = df_optimized[col].astype('category')
    
    # Save to Parquet
    test_file = 'test_integration.parquet'
    df_optimized.to_parquet(test_file, compression='snappy', index=False)
    
    file_size = os.path.getsize(test_file)
    print(f"\nSaved to Parquet: {file_size} bytes")
    
    # Load back
    df_loaded = pd.read_parquet(test_file)
    print(f"Loaded from Parquet: {len(df_loaded)} rows")
    
    # Verify data integrity
    assert len(df_loaded) == len(df), "Row count mismatch!"
    
    # Check precision (float32 should be within tolerance)
    l2_diff = (df['l2_distance'] - df_loaded['l2_distance']).abs().max()
    cos_diff = (df['cosine_distance'] - df_loaded['cosine_distance']).abs().max()
    
    print(f"\nPrecision check:")
    print(f"  L2 distance max diff: {l2_diff:.6e} (should be < 0.0001)")
    print(f"  Cosine distance max diff: {cos_diff:.6e} (should be < 0.0001)")
    
    assert l2_diff < 0.0001, f"L2 precision loss too large: {l2_diff}"
    assert cos_diff < 0.0001, f"Cosine precision loss too large: {cos_diff}"
    
    # Cleanup
    os.remove(test_file)
    
    print("\n[PASS] All tests passed!")
    print("\nParquet integration is working correctly!")
    print("Ready to use with mask_impact_vl.py")

if __name__ == '__main__':
    try:
        test_parquet_roundtrip()
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("\nPlease install required packages:")
        print("  pip install pyarrow>=14.0.0")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

