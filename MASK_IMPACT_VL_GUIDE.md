# Vision-Language Token Masking Analysis Guide

Complete guide for analyzing token masking impact in Vision-Language models using `mask_impact_vl.py`.

## Overview

This tool measures how masking individual tokens from attention K/V affects the residual stream updates at each layer in Vision-Language models (Qwen3-VL). It supports both text-only and multimodal (image + text) analysis.

## Features

- **Multimodal Support**: Analyze text tokens, vision tokens, or both
- **Batch Processing**: Process multiple prompts from YAML config
- **Efficient Computation**: Batched token masking for 8x+ speed improvement
- **Memory Management**: Automatic CUDA memory management and OOM handling
- **Per-Head Analysis**: Optional per-head attention analysis (can be disabled for 32x speedup)
- **Attention Weights**: Optional attention weight capture for visualization
- **Multiple Output Formats**: CSV (compatibility) + Parquet (99%+ smaller, fast web viewing)
- **Interactive Visualization**: Specialized grid viewer for vision tokens

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Qwen3-VL requires transformers >= 4.57.0
pip install transformers>=4.57.0
```

### Basic Usage

**Text-Only Analysis:**
```bash
python mask_impact_vl.py --prompt "What is the capital of France?" --num-tokens 5
```

**Image + Text Analysis:**
```bash
python mask_impact_vl.py \
  --image images/testimg.png \
  --prompt "What's in this image?" \
  --num-tokens 5 \
  --mask-mode text  # text, vision, or both
```

**Batch Mode (Recommended):**
```bash
python mask_impact_vl.py --config prompts_config_vl.yaml
```

## Masking Modes

### Phase 1: Text-Only
Analyze text prompts without images (validates architecture).

```bash
python mask_impact_vl.py --prompt "The Eiffel Tower is in"
```

### Phase 2a: Image + Text (Masking Text Only)
Analyze which text tokens matter when processing an image.

```bash
python mask_impact_vl.py \
  --image images/cat.jpg \
  --prompt "What animal is this?" \
  --mask-mode text
```

### Phase 2b: Image + Text (Masking Vision Only)
Analyze which image regions matter for the answer.

```bash
python mask_impact_vl.py \
  --image images/cat.jpg \
  --prompt "What animal is this?" \
  --mask-mode vision
```

### Phase 2c: Image + Text (Masking Both)
Comprehensive analysis of all tokens.

```bash
python mask_impact_vl.py \
  --image images/cat.jpg \
  --prompt "What animal is this?" \
  --mask-mode both
```

## Performance Options

### Batch Size

Control memory usage vs speed tradeoff:

```bash
# Default (balanced)
python mask_impact_vl.py --prompt "test" --batch-size 8

# High memory GPU (faster)
python mask_impact_vl.py --prompt "test" --batch-size 16 --device cuda

# Low memory GPU (slower but safer)
python mask_impact_vl.py --prompt "test" --batch-size 4
```

### Skip Per-Head Analysis

**32x speed improvement** by skipping per-head analysis:

```bash
python mask_impact_vl.py --prompt "test" --skip-per-head
```

### Combined for Maximum Speed

```bash
python mask_impact_vl.py \
  --prompt "test" \
  --skip-per-head \
  --batch-size 16 \
  --device cuda
```

## Batch Mode Configuration

Create a YAML config for multiple experiments:

```yaml
# prompts_config_vl.yaml

model:
  path: "Qwen/Qwen3-VL-4B-Instruct"

device: "cuda"  # or "cpu"
batch_size: 8
skip_per_head: false
save_attention_weights: false  # Enable for attention weight visualization
max_image_resolution: 768  # Resize images larger than this

prompts:
  - name: "cat_identification"
    enabled: true
    prompt: |
      <|im_start|>user
      <image>
      What animal is this?<|im_end|>
      <|im_start|>assistant
    image_path: "images/cat.jpg"
    num_tokens: 5
    mask_mode: "text"
    
  - name: "scene_understanding"
    enabled: true
    prompt: |
      <|im_start|>user
      <image>
      Describe what you see<|im_end|>
      <|im_start|>assistant
    image_path: "images/scene.jpg"
    num_tokens: 10
    mask_mode: "both"
```

Run with:
```bash
python mask_impact_vl.py --config prompts_config_vl.yaml
```

## Output Data

### Columns

Each row represents masking a single token:

- `generation_step`: Which output token is being predicted (0 = initial prompt)
- `layer`: Layer index (0 to num_layers-1)
- `token_masked`: The token that was masked
- `token_position`: Position of the masked token in the sequence
- `variant`: Analysis variant (see below)
- `l2_distance`: L2 distance between baseline and masked updates
- `cosine_distance`: Cosine distance (1 - cosine similarity)
- `attention_weight`: Raw attention weight (when save_attention_weights=true)

### Analysis Variants

**Standard Variants:**
- **`Full`**: Complete residual stream update (attention + MLP)
- **`Attn`**: Only the attention component's contribution
- **`Head_N`**: Individual attention head contributions (N = 0 to num_heads-1)

**Attention Weight Variants** (when `save_attention_weights: true`):
- **`AttentionWeight_Head_N`**: Raw attention weight from head N to the masked token
- **`AttentionWeight_Avg`**: Average attention weight across all heads

### Understanding Distance Metrics

**L2 Distance**: Measures total magnitude of change
- Typical range: 0.0 to 100+ (depends on model)
- Higher values = masking this token causes larger changes

**Cosine Distance**: Measures directional change
- Range: 0.0 (same direction) to 2.0 (opposite direction)
- Values typically 100-1000x smaller than L2
- Many values may be ~0.0 (expected for tokens with minimal impact)
- Saved with 6 decimal precision to preserve small values

### File Formats

**Parquet** (recommended):
- 99%+ smaller than CSV
- Fast loading in web viewer
- Compressed with Snappy for browser compatibility
- `output/{name}_results.parquet`

**CSV** (compatibility):
- Opens in spreadsheets
- Larger file size
- `output/{name}_results.csv`

## Visualization

### Standard Table Viewer

Use `visualize_results.html` for tabular data:

1. Open `visualize_results.html` in a browser
2. Load `output/{name}_results.parquet`
3. Filter by layer, step, variant
4. Interactive heatmaps and scaling options

### Vision-Language Grid Viewer

Use `visualize_vl_grid.html` for spatial analysis of image tokens:

1. Open `visualize_vl_grid.html` in a browser
2. Load `output/{name}_results.parquet`
3. Optionally load the original image for overlay
4. View image tokens as 2D grid with spatial patterns
5. Navigate layers/steps with keyboard

**Features:**
- Image tokens displayed as 2D grid (auto-detects dimensions)
- Text tokens displayed separately
- Independent color scaling for each modality
- Keyboard navigation (←/→ for layers, ↑/↓ for steps)
- Image overlay to see which regions are important
- Multiple scaling modes (linear, square root, percentile)

See `VL_GRID_VIEWER_GUIDE.md` for detailed usage.

## Advanced Features

### Attention Weight Visualization

Enable attention weight capture:

```yaml
save_attention_weights: true
```

This adds attention weight variants to the output, allowing you to:
- See which tokens the model attends to
- Compare attention patterns across heads
- Correlate attention with masking impact

View in `visualize_vl_grid.html` by selecting `AttentionWeight_*` variants.

### Image Resolution Control

Resize large images to save memory:

```bash
python mask_impact_vl.py \
  --image large_image.jpg \
  --max-image-resolution 768 \
  --prompt "What's in this image?"
```

Images larger than 768px will be downscaled while maintaining aspect ratio.

### Multiple Generation Steps

Analyze how token importance changes as the model generates:

```bash
python mask_impact_vl.py \
  --prompt "The capital of France is" \
  --num-tokens 5
```

This generates 5 tokens and analyzes token importance at each step.

## Performance Benchmarks

**Speed Improvements:**
- Batched processing (batch_size=8): ~8x faster
- Skip per-head (--skip-per-head): 32x fewer computations
- Combined: Up to 256x faster for quick experiments

**Memory Usage:**
- Text-only: ~2-4 GB VRAM (4B model)
- Image + Text: ~6-10 GB VRAM (depends on image resolution)
- Use --batch-size 4 if encountering OOM errors

**File Sizes:**
- CSV: ~10-100 MB per experiment
- Parquet: ~0.1-2 MB per experiment (99%+ smaller!)

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python mask_impact_vl.py --batch-size 4

# Resize images
python mask_impact_vl.py --max-image-resolution 512

# Use CPU (slower but works)
python mask_impact_vl.py --device cpu
```

### Import Errors

```bash
# Update transformers
pip install transformers>=4.57.0

# Install missing dependencies
pip install -r requirements.txt
```

### Slow Processing

```bash
# Skip per-head analysis
python mask_impact_vl.py --skip-per-head

# Increase batch size (if memory allows)
python mask_impact_vl.py --batch-size 16
```

## Example Workflows

### Quick Experiment

```bash
python mask_impact_vl.py \
  --image images/testimg.png \
  --prompt "What is this?" \
  --num-tokens 3 \
  --skip-per-head \
  --batch-size 16
```

### Comprehensive Analysis

```bash
python mask_impact_vl.py \
  --image images/scene.jpg \
  --prompt "Describe this scene in detail" \
  --num-tokens 20 \
  --mask-mode both \
  --save-attention-weights \
  --output detailed_scene_analysis
```

### Batch Production Run

```bash
# Create config with 10+ experiments
python mask_impact_vl.py --config production_experiments.yaml
```

## Related Documentation

- **`BATCH_MODE_GUIDE.md`** - Detailed batch configuration guide
- **`VL_GRID_VIEWER_GUIDE.md`** - Vision-language visualization guide
- **`README.md`** - Project overview
- **`Qwen3-VL.md`** - Model architecture reference

## Citation

If you use this tool in research, please cite the Qwen3-VL paper and acknowledge this analysis framework.

---

**Created for token-level attention analysis in Vision-Language models**

