# Parquet File Splitting - Summary

## Problem
The original `mask_impact_vl.py` script was concatenating all prompts into a single parquet file, making it difficult to work with individual prompts in the visualization tool.

## Solution

### 1. Updated `mask_impact_vl.py`
**Changes made:**
- Each prompt is now saved as a **separate file** during processing
- Format: `{config_name}_{prompt_name}.parquet` and `.csv`
- No more concatenated files when using config mode
- Better compression with optimized data types
- Shows file sizes during save

**Behavior:**
- **Config mode** (YAML file): Saves each prompt separately
  - Example: `image_analysis_orb_looking_at.parquet`
- **Command-line mode**: Still saves a single file (unchanged)

### 2. Created `split_parquet_by_prompt.py`
**Purpose:** Split existing concatenated parquet files into separate files per prompt

**Usage:**
```bash
# Split a single file
python split_parquet_by_prompt.py output/eras_text_orb_looking_at.parquet

# Split multiple files
python split_parquet_by_prompt.py output/*.parquet

# Dry run (preview without saving)
python split_parquet_by_prompt.py --dry-run output/*.parquet

# Specify output directory
python split_parquet_by_prompt.py --output-dir output/split file.parquet
```

**Features:**
- Auto-detects prompts using `prompt_name` column
- Optimizes data types for compression
- Shows file sizes and compression ratios
- Asks before deleting original file
- Handles multiple files in one run

## Results

### Your Split Files
From `eras_text_orb_looking_at.parquet` (17.8 MB) → 4 separate files:

| File | Rows | CSV Size | Parquet Size | Grid |
|------|------|----------|--------------|------|
| eras_text | 6,774,840 | 1,044 MB | 6.92 MB | ~23×23 |
| eras_figures | 4,738,104 | 789 MB | 4.83 MB | ~23×23 |
| eras_legs | 1,345,176 | 209 MB | 1.37 MB | ~23×23 |
| **orb_looking_at** | **2,192,184** | **342 MB** | **2.23 MB** | **24×24 ✓** |

### Verification
✅ **orb_looking_at** file confirmed:
- Only contains `'orb_looking_at'` prompt
- 576 vision tokens (perfect 24×24 grid)
- Proper token positions
- Ready for visualization

## Using with Visualization Tool

Now when you use `visualize_vl_grid.html`:
1. Load `eras_text_orb_looking_at_orb_looking_at.parquet`
2. You'll see ONLY the orb_looking_at prompt data
3. Perfect 24×24 grid will be displayed
4. No confusion from mixed prompts!

Each other prompt can be loaded separately for individual analysis.

## Next Steps

### For Future Runs
1. Use YAML config mode (already set up)
2. Each prompt automatically saves separately
3. No need to split files manually

### For Existing Files
1. Run `split_parquet_by_prompt.py` on any concatenated files
2. Optionally delete originals after verifying splits
3. Load individual prompts in visualization tool

## File Locations
- **Split script**: `split_parquet_by_prompt.py`
- **Updated script**: `mask_impact_vl.py`
- **Split files**: `output/eras_text_orb_looking_at_*.parquet`
- **Original file**: `output/eras_text_orb_looking_at.parquet` (kept as backup)
- **Analysis scripts**: `analyze_*.py`, `check_*.py` (for debugging)

