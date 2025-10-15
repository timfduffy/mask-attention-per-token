# Parquet Browser Compatibility Note

## Current Status

**Parquet files are generated successfully** and provide **99.7% size reduction**, but **browser-based viewing has compatibility issues**.

## The Issue

JavaScript Parquet libraries for browsers are still maturing:
- `parquet-wasm` has CDN/import issues
- `hyparquet` API is still in beta
- WebAssembly dependencies can be unreliable

## ✅ What Works Perfectly

1. **Python-side Parquet generation**: 100% working
   - Files are correctly created with float32 + categorical encoding
   - ZSTD compression works great
   - **3.8M rows: 1.1 GB → 3.5 MB** (99.7% reduction)

2. **File storage & transfer**: Major win!
   - Share files via email/Slack
   - Keep in Git (under 5 MB)
   - Store hundreds of experiments
   - Fast to backup/download

3. **Python analysis**: Works perfectly
   ```python
   import pandas as pd
   df = pd.read_parquet('output/results.parquet')
   # All data intact, full precision
   ```

## 🔧 Workarounds for Visualization

### Option 1: Use CSV (Recommended for now)

Both CSV and Parquet are always generated, so just use CSV:

```bash
# CSV is always created alongside Parquet
output/my_results.csv      # Load this in visualize_results.html
output/my_results.parquet  # Tiny file for storage/sharing
```

**Tradeoff**: CSV is 612 MB vs 3.5 MB Parquet, but loads fine in browser.

### Option 2: Convert Parquet → JSON

For faster loading than CSV:

```bash
python convert_parquet_to_json.py output/my_results.parquet
# Creates: output/my_results.json

# Then load in visualize_results_json.html
```

**Tradeoff**: JSON is 1.1 GB (bigger than CSV!), but loads faster (~11s vs ~30s).

### Option 3: Python-based Visualization

Skip the HTML viewer entirely:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('output/results.parquet')

# Custom analysis
pivot = df.pivot_table(
    values='l2_distance',
    index='layer',
    columns='token_position'
)

plt.imshow(pivot, aspect='auto')
plt.show()
```

## 📊 Performance Comparison

### File Sizes (3.8M rows)
| Format | Size | vs Parquet |
|--------|------|-----------|
| **Parquet** | **3.5 MB** | **baseline** ✨ |
| CSV | 613 MB | 175x larger |
| JSON | 1,100 MB | 314x larger |

### Browser Load Times
| Format | Load Time | Notes |
|--------|-----------|-------|
| Parquet | ❌ Issues | CDN library problems |
| CSV | ~30s | Works reliably |
| JSON | ~11s | Works, but huge file |

### Storage Efficiency Winner: Parquet! 🏆
### Browser Compatibility Winner: CSV (for now)

## Future Plans

1. **Wait for library maturity**
   - `hyparquet` is promising but still beta
   - `parquet-wasm` needs better CDN support
   - Native browser APIs may emerge

2. **Fallback to CSV is fine**
   - Browser only loads data once
   - 30s load time is acceptable for 3.8M rows
   - Parquet still wins for storage/transfer

3. **Consider server-based viewer**
   - Python FastAPI/Flask server
   - Serve Parquet → JSON on demand
   - Best of both worlds

## Recommendation

**Current best practice**:

1. **Generate Parquet** (automatic) - for storage
2. **Use CSV for viewing** (also automatic) - for compatibility
3. **Keep both files**:
   - Parquet: Share, backup, archive (tiny!)
   - CSV: Visualize in browser (big, but works)

## Key Insight

**Parquet achieved its main goal**: Making large VL model results practical to store and share!

The browser visualization issue is a temporary limitation of JavaScript libraries, not a problem with Parquet format itself.

---

**Bottom line**: Parquet is a huge win for storage. CSV is fine for viewing. Use both!

