# Text-Only Token Masking Analysis Guide

Complete guide for analyzing token masking impact in text-only transformer models using `mask_impact_analysis.py`.

## Overview

This tool measures how masking individual tokens from attention K/V affects the residual stream updates at each layer in text-only transformer models (e.g., Qwen3-4B-Instruct). It's optimized for analyzing pure language models.

## Features

- **Text-Only Focus**: Optimized for language models without vision components
- **Batch Processing**: Process multiple prompts from YAML config
- **Efficient Computation**: Batched token masking for 8x+ speed improvement
- **Memory Management**: Automatic CUDA memory management and OOM handling
- **Per-Head Analysis**: Optional per-head attention analysis (can be disabled for 32x speedup)
- **Multiple Output Formats**: CSV (compatibility) + JSON (fast web viewing)
- **Interactive Visualization**: Web-based heatmap viewer with filtering

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Single Prompt:**
```bash
python mask_impact_analysis.py --prompt "What is the capital of France?" --num-tokens 5
```

**From File:**
```bash
python mask_impact_analysis.py --prompt prompt.txt --num-tokens 10
```

**Batch Mode (Recommended):**
```bash
python mask_impact_analysis.py --config prompts_config.yaml
```

## Command-Line Arguments

```bash
python mask_impact_analysis.py [OPTIONS]

Options:
  --config PATH           YAML config for batch processing (overrides other args)
  --prompt TEXT          Prompt text or path to .txt file
  --num-tokens N         Number of tokens to generate (default: 1)
  --output NAME          Output file basename (default: masking_results)
  --model PATH           Model path (default: local Qwen3-4B)
  --device DEVICE        Device: auto, cuda, or cpu (default: auto)
  --batch-size N         Batch size for token masking (default: 8)
  --skip-per-head        Skip per-head analysis (32x speedup)
```

## Performance Options

### Batch Size

Control memory usage vs speed tradeoff:

```bash
# Default (balanced)
python mask_impact_analysis.py --prompt "test" --batch-size 8

# High memory GPU (faster)
python mask_impact_analysis.py --prompt "test" --batch-size 16 --device cuda

# Low memory GPU (slower but safer)
python mask_impact_analysis.py --prompt "test" --batch-size 4
```

### Skip Per-Head Analysis

**32x speed improvement** by skipping per-head analysis:

```bash
python mask_impact_analysis.py --prompt "test" --skip-per-head
```

Only analyzes `Full` and `Attn` variants (skips individual heads).

### Combined for Maximum Speed

```bash
python mask_impact_analysis.py \
  --prompt "test" \
  --skip-per-head \
  --batch-size 16 \
  --device cuda
```

Can achieve **256x speedup** compared to baseline!

## Batch Mode Configuration

Create a YAML config for multiple experiments:

```yaml
# prompts_config.yaml

model:
  path: "Qwen/Qwen3-4B-Instruct"

device: "cuda"  # or "cpu"

prompts:
  - name: "geography"
    enabled: true
    prompt: "The capital of France is"
    num_tokens: 5
    
  - name: "math"
    enabled: true
    prompt: "If x=5 and y=3, then x+y="
    num_tokens: 3
    
  - name: "reasoning"
    enabled: false  # Skip this one
    prompt: "If all cats are animals, and Fluffy is a cat, then"
    num_tokens: 10
```

Run with:
```bash
python mask_impact_analysis.py --config prompts_config.yaml
```

Each enabled prompt generates separate output files:
- `output/geography_results.csv`
- `output/geography_results.json`
- `output/math_results.csv`
- `output/math_results.json`

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

### Analysis Variants

- **`Full`**: Complete residual stream update (attention + MLP)
- **`Attn`**: Only the attention component's contribution
- **`Head_N`**: Individual attention head contributions (N = 0 to num_heads-1)

### Understanding Distance Metrics

**L2 Distance**: Measures total magnitude of change
- Typical range: 0.0 to 100+ (depends on model)
- Higher values = masking this token causes larger changes

**Cosine Distance**: Measures directional change
- Range: 0.0 (same direction) to 2.0 (opposite direction)
- Values typically 100-1000x smaller than L2
- Many values may be ~0.0 (expected for tokens with minimal impact)

**Example Interpretation:**

```
token_masked: "Paris"
l2_distance: 15.7
cosine_distance: 0.145
```
- Masking "Paris" significantly changes the output (high L2)
- The direction also changes substantially (high cosine distance)
- This token is important for prediction

```
token_masked: "the"
l2_distance: 0.3
cosine_distance: 0.002
```
- Masking "the" barely changes the output (low L2)
- Direction remains almost the same (very low cosine distance)
- This token is not important for prediction

### File Formats

**CSV** (compatibility):
- Opens in Excel/spreadsheets
- Easy to filter and analyze
- `output/{name}_results.csv`

**JSON** (web viewer):
- Fast loading in browser
- Used by `visualize_results.html`
- `output/{name}_results.json`

## Visualization

Use `visualize_results.html` for interactive analysis:

1. Open `visualize_results.html` in any web browser
2. Load `output/{name}_results.json` or `.csv`
3. Features:
   - Filter by generation step
   - Select layer to view
   - Choose variant (Full, Attn, Head_N)
   - Toggle metric (L2 vs Cosine)
   - Heatmap coloring (linear or square root scaling)
   - Exclude first token (attention sink)
   - Sticky headers for large datasets

## Advanced Usage

### Multiple Generation Steps

Analyze how token importance changes as the model generates:

```bash
python mask_impact_analysis.py \
  --prompt "The capital of France is" \
  --num-tokens 10
```

This generates 10 tokens and analyzes at each step:
- Step 0: Initial prompt only
- Step 1: Prompt + 1st generated token
- Step 2: Prompt + 2 generated tokens
- etc.

### Custom Model

```bash
python mask_impact_analysis.py \
  --prompt "test" \
  --model "meta-llama/Llama-2-7b-hf" \
  --device cuda
```

### Long Prompts from File

```bash
# Create prompt.txt with your text
echo "Your long prompt here..." > my_prompt.txt

python mask_impact_analysis.py \
  --prompt my_prompt.txt \
  --num-tokens 5
```

### Output to Custom Location

```bash
python mask_impact_analysis.py \
  --prompt "test" \
  --output my_experiment_name
```

Generates:
- `output/my_experiment_name.csv`
- `output/my_experiment_name.json`

## Performance Benchmarks

**Speed Improvements:**
- Batched processing (batch_size=8): ~8x faster than sequential
- Skip per-head (--skip-per-head): 32x fewer computations
- Combined: Up to 256x faster for quick experiments

**Typical Processing Times** (Qwen3-4B, CUDA):
- 10 tokens, 32 layers, per-head: ~2 minutes
- 10 tokens, 32 layers, skip-per-head: ~4 seconds
- 100 tokens, 32 layers, skip-per-head: ~40 seconds

**Memory Usage:**
- CPU: ~8 GB RAM
- CUDA: ~4 GB VRAM (batch_size=8)
- Reduce batch_size if OOM errors occur

**File Sizes:**
- CSV: ~1 MB per 10,000 rows
- JSON: ~0.8 MB per 10,000 rows
- Typical experiment: 10-100 MB

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python mask_impact_analysis.py --batch-size 4

# Use CPU (slower but works)
python mask_impact_analysis.py --device cpu

# Skip per-head analysis
python mask_impact_analysis.py --skip-per-head
```

### Slow Processing

```bash
# Skip per-head for 32x speedup
python mask_impact_analysis.py --skip-per-head

# Increase batch size (if memory allows)
python mask_impact_analysis.py --batch-size 16

# Use GPU
python mask_impact_analysis.py --device cuda
```

### Import Errors

```bash
# Install missing dependencies
pip install -r requirements.txt

# Update transformers
pip install --upgrade transformers
```

## Example Workflows

### Quick Experiment (< 5 seconds)

```bash
python mask_impact_analysis.py \
  --prompt "Paris is the capital of" \
  --num-tokens 2 \
  --skip-per-head \
  --batch-size 16
```

### Detailed Analysis (Full per-head)

```bash
python mask_impact_analysis.py \
  --prompt "In the context of machine learning, attention mechanisms" \
  --num-tokens 10 \
  --batch-size 8 \
  --output attention_analysis
```

### Batch Production Run

```bash
# Create config with multiple experiments
python mask_impact_analysis.py --config production_prompts.yaml
```

### Compare Models

```bash
# Run same prompt on different models
python mask_impact_analysis.py \
  --prompt "The meaning of life is" \
  --model "Qwen/Qwen3-4B" \
  --output qwen_life

python mask_impact_analysis.py \
  --prompt "The meaning of life is" \
  --model "meta-llama/Llama-2-7b" \
  --output llama_life
```

## Understanding Results

### Finding Important Tokens

High L2 distance = token is important for prediction

1. Open `visualize_results.html`
2. Load your results
3. Select layer (later layers often show clearer patterns)
4. Sort by L2 distance (descending)
5. Tokens at the top are most important

### Analyzing by Layer

Token importance varies by layer:

- **Early layers** (0-10): Often show general attention patterns
- **Middle layers** (10-20): Start showing semantic importance
- **Late layers** (20-32): Strongest signal for prediction

### Attention Sinks

The first token often has high masking impact but isn't semantically important. Use "Exclude pos 0" option in visualizer to filter this out.

### Generation Step Analysis

When using `--num-tokens > 1`:

- **Step 0**: Which tokens matter for first prediction
- **Step 1+**: How importance shifts as context grows
- **Pattern**: Often see focus shift to recent tokens

## Related Documentation

- **`BATCH_MODE_GUIDE.md`** - Detailed batch configuration guide
- **`README.md`** - Project overview
- **`MASK_IMPACT_VL_GUIDE.md`** - Vision-language version guide

## Comparison with VL Version

| Feature | Text-Only (`mask_impact_analysis.py`) | VL (`mask_impact_vl.py`) |
|---------|----------------------------------------|---------------------------|
| Input | Text only | Text + images |
| Models | Language models | Vision-language models |
| Output | CSV + JSON | CSV + Parquet |
| Masking | All text tokens | Text, vision, or both |
| Viewer | Table viewer | Grid viewer + table viewer |
| Speed | Faster (simpler input) | Slower (image processing) |

## Tips and Best Practices

1. **Start with --skip-per-head** for quick exploration
2. **Use batch mode** for multiple related experiments
3. **Analyze later layers first** (usually more informative)
4. **Focus on L2 distance** (cosine is often noisy)
5. **Use batch_size=16** on high-memory GPUs for maximum speed
6. **Visualize in browser** rather than analyzing raw CSV
7. **Exclude attention sinks** (position 0) when analyzing semantics

---

**Created for token-level attention analysis in transformer language models**

