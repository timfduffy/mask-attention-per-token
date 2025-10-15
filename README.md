# Token Masking Impact Analysis

Measure how masking individual tokens from attention K/V affects the residual stream updates at each layer.

## Overview

This tool helps understand which tokens are most important at which layers by:
1. Running a baseline forward pass through a transformer layer
2. Running a masked pass where one token is excluded from attention (via -inf masking)
3. Computing distance metrics (L2 and cosine) between baseline and masked residual stream updates

## Three Variants Measured

- **Full**: Complete residual stream update from the layer (attention + MLP)
- **Attn**: Only the attention component's contribution
- **Head_N**: Individual attention head contributions (before combining)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test (Single Layer)

Test on just the first layer to verify everything works:

```bash
python test_single_layer.py
```

### Batch Mode (Recommended)

Process multiple prompts at once using a YAML config file:

```bash
python mask_impact_analysis.py --config prompts_config.yaml
```

Edit `prompts_config.yaml` to:
- Add/remove prompts
- Enable/disable prompts with `enabled: true/false`
- Set different `num_tokens` for each prompt
- See `BATCH_MODE_GUIDE.md` for full documentation

**Benefits**:
- Run overnight experiments with multiple prompts
- Easy to organize and version control your experiments
- Each prompt gets its own output files (named by prompt name)

### Single Prompt Mode

Run a single analysis:

```bash
# From prompt.txt (default)
python mask_impact_analysis.py --num-tokens 10

# Inline prompt
python mask_impact_analysis.py --prompt "The capital of France is" --num-tokens 5

# Custom output name
python mask_impact_analysis.py --prompt "test" --output my_experiment --num-tokens 3
```

**Note**: All output files are saved to the `output/` directory (automatically created).

### Custom Usage

```python
from mask_impact_analysis import run_masking_experiment

# Run with custom prompt (uses local Qwen3-0.6B by default)
df = run_masking_experiment(
    prompt="Your custom prompt here",
    device="cuda"  # or "cpu"
)

# Or specify a different model
df = run_masking_experiment(
    prompt="Your custom prompt here",
    model_name=r"path\to\your\model",
    device="cuda"
)

# Save results
df.to_csv("results.csv", index=False)
```

## Output Format

The analysis generates two output files in the `output/` directory:
- **`output/{name}_results.csv`** - For spreadsheet analysis
- **`output/{name}_results.parquet`** - For fast web visualization (recommended, 99%+ smaller!)

Both contain the same data in long format with the following columns:
- `generation_step`: Which output token is being predicted (0 = initial prompt, 1 = after 1st generated token, etc.)
- `layer`: Layer index (0 to num_layers-1)
- `token_masked`: The token that was masked
- `token_position`: Position of the masked token in the sequence
- `variant`: One of "Full", "Attn", or "Head_N"
- `l2_distance`: L2 distance between baseline and masked updates
- `cosine_distance`: Cosine distance (1 - cosine similarity)

**Note**: When `num_output_tokens > 1`, the analysis runs iteratively:
- Step 0: Analyze initial prompt
- Step 1: Generate 1st token, analyze full sequence (prompt + 1 token)
- Step 2: Generate 2nd token, analyze full sequence (prompt + 2 tokens)
- etc.

## Visualization

### Standard Viewer (`visualize_results.html`)

Use **`visualize_results.html`** for standard table-based visualization:

1. Open `visualize_results.html` in any web browser
2. Load a results file from the `output/` directory:
   - `output/{name}_results.parquet` (recommended - 99%+ smaller, fast loading!)
   - `output/{name}_results.csv` (backup - works everywhere)
   
   **Note**: Parquet uses hyparquet library with Snappy compression for browser compatibility.
3. Features:
   - **Toggle metrics**: Switch between L2 and Cosine distance
   - **Filter by step**: View one generation step at a time (much faster!)
   - **Select variant**: Choose Full, Attn, or any Head
   - **Heatmap colors**: Row-based intensity with linear/square-root scaling
   - **Exclude pos 0**: Toggle to exclude attention sink from color scale
   - **Sticky headers**: Scroll through data while keeping headers visible

**Performance**: Parquet is dramatically faster and smaller. Example (3.8M rows):
- **Parquet (Snappy)**: Fast load, ~4-5 MB (0.7% of CSV) ✨
- JSON: 11.5s load, 1.1 GB (183% of CSV)
- CSV: 20-30s load, 613 MB

**Note for Large Files**: Files with 100K+ rows load quickly, but rendering all data at once can freeze the browser. The viewer automatically defaults to showing **Step 0 only** for performance. Use the step selector to view other steps individually.

**Create a Sample** (for testing):
```bash
python create_sample_parquet.py 10000  # Creates a 10K row sample
```

### Vision-Language Grid Viewer (`visualize_vl_grid.html`) 🆕

Specialized viewer for **image token analysis** with 2D grid layout:

**Features:**
- 🖼️ **Image tokens as grid**: Visualizes image tokens in their natural 2D structure
- 📝 **Text tokens separately**: Shows text tokens in a horizontal layout
- 🎨 **Separate heatmaps**: Image and text tokens colored relative to their own maximums
- ⌨️ **Keyboard navigation**:
  - `←/→` Navigate between layers
  - `↑/↓` Navigate between generation steps
  - `M` Toggle metric (L2 ↔ Cosine)
  - `S` Toggle scaling (Linear ↔ Square Root)
- 📊 **Statistics**: Shows max/avg values for both image and text tokens
- 🎯 **Focused view**: Shows one layer + one step at a time for clarity

**Usage:**
```bash
# Open in browser
open visualize_vl_grid.html

# Load your Parquet file
# Starts at Layer 0, Step 0
# Use arrow keys to navigate
```

**Perfect for:**
- Analyzing spatial patterns in image token attention
- Comparing image vs text token importance
- Exploring layer-by-layer evolution
- Understanding generation step progression

## Early Stopping

Generation automatically stops when the model produces `<|im_end|>` or EOS token:
- Set generous `num_tokens` values (e.g., 50) without worrying
- The script stops as soon as the model finishes
- Saves computation time and keeps analysis focused on real output

## Implementation Details

### Performance Optimizations

The implementation includes important optimizations to reduce computational cost:

**KV Cache Reuse**: For each layer L, we pre-compute and cache the hidden states at the input to that layer. This means:
- Layers 0 to L-1 are computed only **once** (during pre-computation)
- For masking experiments at layer L, we reuse these cached states
- Only layer L itself is recomputed (once for baseline, N times for each masked token)

**Complexity Reduction**:
- Without optimization: O(L² × N) layer computations
- With optimization: O(L) + O(L × N) layer computations
- For 28 layers and 5 tokens: ~3,920 → ~168 layer computations (23× speedup!)

### Masking Approach

We mask tokens by setting attention logits to `-inf` before softmax. This ensures:
- The masked token receives exactly 0 attention weight
- Other tokens' weights are renormalized without it
- True removal from attention computation (not just zeroing K/V)

### Measurements

All measurements focus on the **final token position**, as this is typically most important for next-token prediction.

For each variant:
- **L2 Distance**: `||Δ_baseline - Δ_masked||_2` - measures magnitude of change
- **Cosine Distance**: `1 - cos(Δ_baseline, Δ_masked)` - measures directional change

## Device Support

Works on both CPU and GPU. For initial testing, use CPU:

```python
df = run_masking_experiment(prompt, device='cpu')
```

For faster processing with larger models, use GPU:

```python
df = run_masking_experiment(prompt, device='cuda')
```

## Expected Insights

This analysis can reveal:
- Which tokens are most important at which layers
- Whether importance is magnitude-based (L2) or directional (cosine)
- How attention vs MLP contributions differ
- Which attention heads are most affected by specific tokens

## Vision-Language Model Support (NEW! 🎉)

**Qwen3-VL Phase 2b** is now available - mask vision tokens to analyze image patches!

```bash
# Phase 1: Text-only
python mask_impact_vl.py --prompt "What is the capital of France?" --num-tokens 5

# Phase 2a: Image + text (masking text only)
python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --num-tokens 5

# Phase 2b: Image + text (masking vision tokens) - NEW!
python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --mask-mode vision --num-tokens 3

# Phase 2c: Mask both text and vision
python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --mask-mode both --num-tokens 1

# Run test suite
python test_vl_phase1.py
```

See [`VL_IMPLEMENTATION.md`](VL_IMPLEMENTATION.md) for:
- Phase 1: Text-only analysis (✅ Complete)
- Phase 2a: Image + text masking - text only (✅ Complete)
- Phase 2b: Image + text masking - vision only (✅ Complete)
- Phase 2c: Image + text masking - both (✅ Complete)
- Phase 3: Vision encoder analysis (📋 Future)

## Project Structure

```
mask_impact/
├── mask_impact_analysis.py      # Text-only models (Qwen3, Llama, etc.)
├── mask_impact_vl.py            # Vision-language models (Qwen3-VL Phase 1) 🆕
├── prompts_config.yaml           # Batch mode for text models
├── prompts_config_vl.yaml       # Batch mode for VL models (text-only) 🆕
├── visualize_results.html        # Interactive results viewer (Parquet) 🆕
├── visualize_vl_grid.html        # VL Grid viewer (2D image tokens) 🆕
├── visualize_results_json.html   # Legacy viewer (JSON)
├── BATCH_MODE_GUIDE.md          # Detailed batch mode documentation
├── VL_IMPLEMENTATION.md         # VL-specific documentation 🆕
├── test_vl_phase1.py            # VL test suite 🆕
├── output/                       # All results saved here
│   ├── {name}_results.csv       # For spreadsheets
│   └── {name}_results.parquet   # For web visualization (99% smaller!) 🆕
└── prompt.txt                    # Single prompt (optional)
```

## Future Extensions

- Masking multiple tokens simultaneously
- Tracking position-specific effects across all positions
- Comparing accumulated effects across layers
- Cross-attention analysis for encoder-decoder models

