# Parquet File Splitting - Progress Report

## Task
Split all parquet files in `output/` folder that contain multiple prompts.

## Status: IN PROGRESS

### Completed (2/12 files)
✅ **cat_eyes_question.parquet** → Split into 2 files:
  - `cat_eyes_question_cat_eyes_question.parquet` (15.72 MB, 16M rows)
  - `cat_eyes_question_cat_feeling_question.parquet` (4.77 MB, 4.8M rows)

✅ **cowbos_describe_clouds.parquet** → Split into 3 files:
  - `cowbos_describe_clouds_claude_describe.parquet` (15.07 MB, 14.7M rows)
  - `cowbos_describe_clouds_cowboy_describe.parquet` (11.80 MB, 11.2M rows)
  - `cowbos_describe_clouds_cowboy_describe_clouds.parquet` (11.81 MB, 11.3M rows)

### In Progress (10 files running in background)
⏳ cowboy_describe.parquet (2 prompts, 25.9M rows)
⏳ cowboy_horse_type.parquet (5 prompts, 41.6M rows)
⏳ cowboy_location.parquet (4 prompts, 38.9M rows)
⏳ eras_text.parquet (7 prompts, 57.2M rows) ⚠️ LARGEST
⏳ eras_text_eras_figures.parquet (2 prompts, 11.5M rows)
⏳ eras_text_eras_legs.parquet (3 prompts, 12.9M rows)
⏳ eras_text_orb_looking_at.parquet (4 prompts, 15.1M rows)
⏳ eras_text_ships_cat_location.parquet (5 prompts, 18.0M rows)
⏳ eras_text_table_legs.parquet (6 prompts, 19.5M rows)
⏳ eras_text_the_mask_content.parquet (7 prompts, 57.2M rows) ⚠️ LARGEST

## Expected Output Files

When complete, you will have individual files for each prompt:

### Cowboy Series
- cowboy_describe_claude_describe.parquet
- cowboy_describe_cowboy_describe.parquet
- cowboy_horse_type_cowboy_location.parquet
- cowboy_horse_type_cowboy_horse_type.parquet
- cowboy_location_cowboy_location.parquet
- (and more...)

### Eras Series
- eras_text_eras_text.parquet (already exists from previous split)
- eras_text_eras_figures.parquet (will be split further)
- eras_text_eras_legs.parquet (will be split further)
- eras_text_orb_looking_at.parquet (already split previously)
- eras_text_ships_cat_location.parquet
- eras_text_table_legs.parquet
- eras_text_the_mask_content.parquet

## Monitoring Progress

### Check if process is still running:
```powershell
Get-Process python | Where-Object {$_.CPU -gt 1}
```

### Count new split files:
```powershell
dir output\*_*.parquet | measure
```

### Check most recently created files:
```powershell
dir output\*.parquet | sort LastWriteTime -Descending | select -First 10
```

### Re-check which files still need splitting:
```bash
python check_all_parquets.py
```

## Estimated Time
- Small files (< 20M rows): ~2-5 minutes each
- Medium files (20-40M rows): ~5-10 minutes each  
- Large files (> 40M rows): ~10-20 minutes each

**Total estimated time: 1-2 hours** for all remaining files

## Next Steps

1. Wait for background process to complete
2. Run `python check_all_parquets.py` to verify all files are split
3. Optionally delete original concatenated files to save space
4. Use individual prompt files in visualization tool

## Original Files (Kept as Backup)
All original concatenated files are preserved in `output/` folder.
You can delete them after verifying the split files are correct.

## Updated Scripts
- ✅ `split_parquet_by_prompt.py` - Now supports `--keep-original` flag
- ✅ `check_all_parquets.py` - Identifies which files need splitting
- ✅ `mask_impact_vl.py` - Updated to save separate files by default

