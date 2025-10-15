# Vision-Language Model Support (Qwen3-VL)

Phase 2a implementation for analyzing token masking in Qwen3-VL models with image inputs.

## Phase 1: Text-Only Analysis ✅ (Complete)

**Goal**: Validate that the masking analysis works with Qwen3-VL architecture using text-only inputs.

### What's Working

- ✅ Model loading with `Qwen3VLForConditionalGeneration`
- ✅ Text processing via `AutoProcessor`
- ✅ Layer access (`model.model.layers` for text decoder)
- ✅ MRoPE handling (text-only uses standard section)
- ✅ Same masking logic as text-only models
- ✅ Multi-token generation with early stopping
- ✅ Output to JSON/CSV in `output/` directory

### Architecture Differences from Text-Only

| Component | Text-Only (Qwen3) | VL (Qwen3-VL) |
|-----------|------------------|---------------|
| Model Class | `AutoModelForCausalLM` | `Qwen3VLForConditionalGeneration` |
| Tokenizer | `AutoTokenizer` | `AutoProcessor` (includes tokenizer) |
| Layers | 28 (Qwen3-4B) | 36 (Qwen3-VL-4B text decoder) |
| Hidden Size | 2048 | 2560 |
| RoPE | Standard | MRoPE (3 sections: text, spatial, temporal) |
| Input | Text only | Text + Vision (Phase 1: text only) |

### Usage

```bash
# Single text prompt
python mask_impact_vl.py --prompt "What is the capital of France?" --num-tokens 5

# From file
python mask_impact_vl.py --prompt prompt.txt --num-tokens 10

# With GPU
python mask_impact_vl.py --prompt "test" --device cuda

# Custom output name
python mask_impact_vl.py --prompt "test" --output phase1_test --num-tokens 3
```

### Current Limitations (Phase 1)

- ❌ No image inputs (text-only for now)
- ❌ No vision token masking
- ❌ No vision encoder analysis
- ❌ No batch mode (single prompts only)

---

## Phase 2a: Image + Text Masking (Text Tokens Only) ✅ (Complete)

**Goal**: Add image inputs and analyze text token masking in multimodal context.

### What's Working

- ✅ Image input via `--image` flag
- ✅ PIL image loading and processing
- ✅ Vision token detection (automatic via `<|vision_start|>` and `<|vision_end|>`)
- ✅ Proper VL attention mask:
  - Vision tokens: Bidirectional attention
  - Text tokens: Causal attention
  - Text can attend to all vision tokens
- ✅ Text-only masking (skips vision tokens)
- ✅ Multi-token generation with images
- ✅ Cross-modal analysis (how text context affects image understanding)

### Architecture: Attention Mask

```python
def create_vl_attention_mask(seq_length, vision_ranges, device):
    """
    Vision tokens: bidirectional (can attend to each other)
    Text tokens: causal (autoregressive)
    Text can attend to vision (vision comes first)
    """
    mask = torch.triu(...)  # Start with causal
    for start, end in vision_ranges:
        mask[start:end, start:end] = 0.0  # Bidirectional vision
        mask[end:, start:end] = 0.0       # Text can see vision
    return mask
```

### Usage

```bash
# Image + text (masking text tokens only)
python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --num-tokens 5

# Compare with text-only
python mask_impact_vl.py --prompt "What's in this image?" --num-tokens 5 --output text_only
python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --num-tokens 5 --output with_image

# With GPU
python mask_impact_vl.py --image photo.jpg --prompt "Describe this" --device cuda
```

### What You Can Learn

- Which text tokens matter most when image is present
- How question words influence vision processing
- Cross-modal attention patterns across layers
- Whether image changes text token importance

### Current Limitations (Phase 2a)

- ❌ No vision token masking (text tokens only)
- ❌ No batch mode
- ❌ No per-patch vision analysis

---

## Phase 2b: Vision Token Masking ✅ (Complete)

**Goal**: Mask individual vision tokens to analyze which image regions matter most.

### What's Working

- ✅ Vision token detection (from Phase 2a)
- ✅ Mask individual vision tokens
- ✅ Measure impact on text generation
- ✅ `--mask-mode` flag: 'text', 'vision', or 'both'
- ✅ Spatial analysis enabled (patch-level importance)

### Implementation

```python
# Controlled by --mask-mode parameter
for mask_pos in range(current_num_tokens):
    is_vision_token = any(start <= mask_pos < end for start, end in vision_ranges)
    
    if mask_mode == 'text' and is_vision_token:
        continue  # Phase 2a: skip vision
    elif mask_mode == 'vision' and not is_vision_token:
        continue  # Phase 2b: skip text
    # mask_mode == 'both': don't skip (Phase 2c)
    
    # Mask this token...
```

### Usage

```bash
# Phase 2b: Mask vision tokens only
python mask_impact_vl.py \
  --image testimg.png \
  --prompt "What's in this image?" \
  --mask-mode vision \
  --num-tokens 3

# Phase 2c: Mask both text and vision
python mask_impact_vl.py \
  --image testimg.png \
  --prompt "What's in this image?" \
  --mask-mode both \
  --num-tokens 1
```

### What You Can Learn

- **Vision token masking**: Which image patches matter most for generation
- **Spatial importance**: Identify critical image regions
- **Cross-modal dependencies**: Do certain patches trigger specific words?
- **Patch granularity**: 32×32 effective pixel regions per token
- **Layer-wise patterns**: When does vision matter most?

### Current Limitations (Phase 2b)

- ❌ No automatic spatial heatmap visualization (CSV only)
- ❌ Large images (1024×1024) are slow (1024 vision tokens!)
- ❌ No batch mode
- ❌ Post-processing needed to map tokens to image coordinates

---

## Phase 3: Vision Encoder Analysis (Advanced)

**Goal**: Analyze masking at the vision encoder level (pre-fusion).

### Architecture Deep Dive

```
Vision Encoder (ViT)
├── 24 layers (depth)
├── 1024 hidden size
├── Patch size: 16x16
├── DeepStack: Fuses multi-level features
└── Output: Visual embeddings → Text decoder

Text Decoder
├── 36 layers
├── 2560 hidden size
├── Receives vision embeddings at specific layers
└── MRoPE for multimodal position encoding
```

### Implementation Approach

1. **Hook into Vision Encoder**
   ```python
   # Access vision model
   vision_model = model.visual
   
   # Hook specific layers (DeepStack visual indexes)
   deepstack_indexes = [5, 11, 17]  # From config
   
   # Mask patches at encoder level
   # Requires different masking logic (spatial, not causal)
   ```

2. **Patch-Level Masking**
   - Mask individual image patches (16x16 pixels)
   - Measure impact on final text generation
   - Create spatial heatmaps of important regions

3. **Layer-wise Analysis**
   - Compare early vs. late vision layers
   - Track how visual features transform
   - Understand DeepStack fusion points

---

## Testing Strategy

### Phase 1 Testing ✅

1. **Basic Functionality**
   ```bash
   # Test model loading
   python mask_impact_vl.py --prompt "Hello world" --num-tokens 1
   
   # Test multi-token
   python mask_impact_vl.py --prompt "Count to five:" --num-tokens 10
   
   # Test early stopping
   python mask_impact_vl.py --prompt "Say 'done':" --num-tokens 50
   ```

2. **Compare with Text-Only**
   - Run same prompt on both `mask_impact_analysis.py` and `mask_impact_vl.py`
   - Verify similar masking patterns (since both are text-only)
   - Check if 36 layers show similar hierarchical patterns

3. **Performance Benchmarks**
   - Time per layer analysis
   - Memory usage (2560 vs 2048 hidden size)
   - GPU vs CPU comparison

### Phase 2 Testing (Future)

1. **Image Loading**
   - Test with simple images (cat, dog, etc.)
   - Verify vision tokens appear in sequence
   - Check token position mapping

2. **Multimodal Prompts**
   - Image + "Describe this"
   - Image + "What color is X?"
   - Multiple images in sequence

3. **Masking Validation**
   - Mask vision tokens → text generation changes?
   - Mask text tokens → image understanding changes?

---

## File Structure

```
mask_impact/
├── mask_impact_analysis.py      # Text-only models (Qwen3, etc.)
├── mask_impact_vl.py            # VL models Phase 1 (text-only) ✅
├── prompts_config.yaml           # Text-only batch config
├── prompts_config_vl.yaml       # VL text-only batch config ✅
├── VL_IMPLEMENTATION.md         # This file ✅
├── output/                       # All results
│   ├── vl_masking_results.csv
│   └── vl_masking_results.json
└── visualize_results.html        # Works with VL outputs too!
```

---

## Key Insights from Qwen3-VL Architecture

### MRoPE (Multi-Resolution RoPE)
- **3 Sections**: `[24, 20, 20]` for text, height, width
- **Text-only**: Uses first 24 dimensions (standard RoPE)
- **With images**: All 64 dimensions (24+20+20)
- **Interleaved**: `mrope_interleaved: true` in config

### DeepStack Integration
- Vision features injected at layers: **5, 11, 17**
- Multi-level ViT features → Text decoder
- Improves image-text alignment

### Token Structure (with images)
```
[vision_start] [img_token_1] ... [img_token_N] [vision_end] [text_tokens...]
      151652        151655              151655        151653      [regular tokens]
```

---

## Troubleshooting

### Common Issues (Phase 1)

1. **Model Loading Fails**
   - Check model path is correct
   - Ensure `transformers>=4.57.0` (for Qwen3-VL support)
   - Try: `pip install git+https://github.com/huggingface/transformers`

2. **MRoPE Errors**
   - Text-only should work with standard RoPE section
   - If issues, check `rope_scaling` config
   - Verify `position_embeddings` shape

3. **Memory Issues**
   - 36 layers + 2560 hidden = larger memory footprint
   - Use `device='cpu'` for testing
   - Enable gradient checkpointing if needed (future)

### Debug Commands

```bash
# Minimal test
python mask_impact_vl.py --prompt "Hi" --num-tokens 1

# Check config
python -c "from transformers import Qwen3VLForConditionalGeneration; \
           m = Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-4B-Instruct'); \
           print(m.config.text_config)"

# Verify layers
python -c "from transformers import Qwen3VLForConditionalGeneration; \
           m = Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-4B-Instruct'); \
           print(f'Layers: {len(m.model.layers)}')"
```

---

## Next Steps

1. **✅ Phase 1 Complete**: Text-only analysis working
2. **→ Phase 2 Next**: Add image inputs, text masking only
3. **→ Phase 3 Future**: Vision token masking
4. **→ Phase 4 Advanced**: Vision encoder analysis

Ready to test Phase 1! 🚀

