# Qwen3-VL Quick Start Guide

Phase 1: Text-only analysis on vision-language models

## Installation

### 1. Install Latest Transformers

Qwen3-VL requires `transformers >= 4.57.0` (currently in development):

```bash
# Install from source (recommended for Qwen3-VL)
pip install git+https://github.com/huggingface/transformers

# Or wait for official release
# pip install transformers==4.57.0  # when available
```

### 2. Verify Installation

```bash
python test_vl_phase1.py
```

This will run 5 tests:
- ✓ Model loading
- ✓ Text processing
- ✓ Forward pass
- ✓ Layer access
- ✓ RoPE embeddings

**Expected output**: All tests pass ✅

## Quick Examples

### Example 1: Simple Question

```bash
python mask_impact_vl.py --prompt "What is 2+2?" --num-tokens 3
```

**What happens:**
1. Loads Qwen3-VL-4B-Instruct
2. Tokenizes text-only prompt
3. Analyzes 36 layers (vs 28 in standard Qwen3)
4. Measures impact of masking each token
5. Saves to `output/vl_masking_results.json`

### Example 2: From File

Create `prompt_vl.txt`:
```
Explain gravity in simple terms.
```

Run:
```bash
python mask_impact_vl.py --prompt prompt_vl.txt --num-tokens 20
```

### Example 3: GPU Acceleration

```bash
python mask_impact_vl.py \
  --prompt "The Eiffel Tower is in" \
  --num-tokens 5 \
  --device cuda \
  --output eiffel_test
```

**Saves to:**
- `output/eiffel_test.csv`
- `output/eiffel_test.json`

### Example 4: Compare Text vs VL Model

**Same prompt, both models:**

```bash
# Text-only model (Qwen3-4B, 28 layers)
python mask_impact_analysis.py \
  --prompt "What is AI?" \
  --num-tokens 10 \
  --output text_model_ai

# VL model (Qwen3-VL-4B, 36 layers)
python mask_impact_vl.py \
  --prompt "What is AI?" \
  --num-tokens 10 \
  --output vl_model_ai
```

**Then compare in visualizer:**
- Load `output/text_model_ai.json`
- Load `output/vl_model_ai.json` 
- See how extra layers affect token importance!

## Visualizing Results

All VL outputs are compatible with the existing visualizer:

```bash
# Open visualize_results.html in browser
# Load: output/vl_masking_results.json
```

**VL-specific features to check:**
- 36 layers instead of 28
- Larger hidden size (2560 vs 2048)
- Similar attention patterns (since text-only)

## Understanding Output

### CSV/JSON Structure

```csv
generation_step,layer,token_masked,token_position,variant,l2_distance,cosine_distance
0,0,"What",0,Full,0.245,0.012
0,0,"What",0,Attn,0.123,0.008
0,0,"What",0,Head_0,0.034,0.003
...
```

**Key differences from text models:**
- More layers (0-35 instead of 0-27)
- Same variants: Full, Attn, Head_0 to Head_31

### Interpretation

**Phase 1 Insights:**
- Which tokens matter at which layers (text-only baseline)
- How VL architecture affects text processing
- Comparison with pure text models

**Phase 2 will add:**
- How images affect token importance
- Cross-modal attention patterns
- Vision token masking

## Troubleshooting

### Issue 1: Model Download Fails

**Error:** `OSError: We couldn't connect to ...`

**Solution:**
```bash
# Pre-download model
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct

# Or use local path
python mask_impact_vl.py --model /path/to/local/model --prompt "test"
```

### Issue 2: Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# 1. Use CPU
python mask_impact_vl.py --device cpu --prompt "test"

# 2. Reduce tokens
python mask_impact_vl.py --num-tokens 1 --prompt "test"

# 3. Use smaller model (future: Qwen3-VL-0.6B)
```

### Issue 3: Wrong Transformers Version

**Error:** `ImportError: cannot import name 'Qwen3VLForConditionalGeneration'`

**Solution:**
```bash
# Install from source
pip install --upgrade git+https://github.com/huggingface/transformers

# Verify
python -c "from transformers import Qwen3VLForConditionalGeneration; print('OK')"
```

### Issue 4: Slow Performance

**36 layers = more computation!**

**Optimizations:**
1. **Use GPU**: `--device cuda` (3-5x faster)
2. **Limit tokens**: `--num-tokens 1` (for testing)
3. **Pre-computed cache**: Already implemented ✅
4. **Batch mode**: Coming in Phase 1.1

## Performance Benchmarks

Approximate times (Qwen3-VL-4B-Instruct):

| Setup | Tokens | Layers | Time | Memory |
|-------|--------|--------|------|--------|
| CPU, 1 token | 10 | 36 | ~2 min | 8 GB |
| CPU, 5 tokens | 10 | 36 | ~8 min | 8 GB |
| GPU (A100), 1 token | 10 | 36 | ~20 sec | 12 GB |
| GPU (A100), 5 tokens | 10 | 36 | ~1.5 min | 12 GB |

**Note:** 36 layers vs 28 = ~30% longer than text-only model

## Next Steps

### ✅ You've completed Phase 1 if:
- Test suite passes
- Can run single prompts
- Outputs visualize correctly

### → Ready for Phase 2 when:
- Phase 1 results validate your hypothesis
- You want to add image inputs
- You need multimodal analysis

### 🚀 Phase 2 Preview

```python
# Coming soon: Image + text analysis
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "cat.jpg"},
            {"type": "text", "text": "What animal is this?"}
        ]
    }
]
```

## Support

- **Issues**: Check `VL_IMPLEMENTATION.md`
- **Text-only models**: Use `mask_impact_analysis.py`
- **Batch processing**: Coming in Phase 1.1
- **Image inputs**: Coming in Phase 2

Happy analyzing! 🔬

