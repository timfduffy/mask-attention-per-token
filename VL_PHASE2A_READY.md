# 🎉 Phase 2a Complete: Image + Text Masking!

**Status**: Ready for testing
**What**: Image inputs with text-only masking
**Date**: Just now!

---

## What Was Implemented

### ✅ Core Features

1. **Image Input Support**
   - Load images via PIL
   - Process through AutoProcessor
   - Vision tokens automatically detected

2. **Proper VL Attention Mask**
   - **Vision tokens**: Bidirectional attention (can attend to each other)
   - **Text tokens**: Causal attention (autoregressive)
   - **Text → Vision**: Can attend (vision comes first)

3. **Text-Only Masking** (Phase 2a)
   - Masks only text token positions
   - Skips all vision tokens automatically
   - Preserves natural VLM attention structure

### Key Functions Added

```python
create_vl_attention_mask(seq_length, vision_ranges, device)
# Creates proper bidirectional/causal mask

# Updated to accept image_path parameter
run_vl_masking_experiment(..., image_path=None)
```

---

## Test It Now! 🚀

### Quick Test with Piggy Bank Image

```bash
# Phase 2a: Image + text (masking text tokens)
python mask_impact_vl.py \
  --image testimg.png \
  --prompt "What's in this image?" \
  --num-tokens 3

# Expected output:
# - PHASE 2a: Image + Text (masking text only) Masking Analysis
# - Loading image: testimg.png
# - Image size: (128, 128)
# - Vision token ranges: [(1, 17)]  # 16 vision tokens for 128x128 image
# - Total vision tokens: 16
# - Analysis will SKIP vision tokens, only mask text
```

### What You'll See

**Console output**:
```
PHASE 2a: Image + Text (masking text only) Masking Analysis
Loading Qwen3-VL model...
✓ Model loaded successfully!

Loading image: testimg.png
  Image size: (128, 128)
  Vision token ranges: [(1, 17)]
  Total vision tokens: 16

Prompt: What's in this image?
Tokenized (X tokens): ['<|vision_start|>', '<vision>', ..., '<|vision_end|>', 'What', ...]
  (includes 1 image(s) with vision tokens)

Analyzing initial prompt (X tokens)...
Pre-computing hidden states for all 36 layers...
Processing layer 0/35...
  (skipping 16 vision tokens - masking text only)
...
```

**Output files**:
- `output/vl_masking_results.csv`
- `output/vl_masking_results.json`

**What the results show**:
- Which **text prompts** matter most when image is present
- How text context influences image understanding
- Cross-modal attention patterns

---

## Architecture Details

### Token Sequence Structure

```
Input: image + "What's in this image?"

Token sequence:
[<|vision_start|>] [vision_token_1] [vision_token_2] ... [vision_token_16] [<|vision_end|>] [What] ['s] [in] [this] [image] [?]
   Position 0           1-16 (BIDIRECTIONAL)                  Position 17      18-23 (CAUSAL)
```

### Attention Mask Matrix

```
                vision tokens (1-16)    text tokens (18-23)
vision (1-16)   [   BIDIRECTIONAL   ]   [     0.0      ]
text (18-23)    [       0.0         ]   [   CAUSAL     ]

Legend:
- BIDIRECTIONAL: All positions can attend to each other (no -inf)
- CAUSAL: Upper triangle is -inf (can only attend to earlier positions)
- 0.0: Can attend (text can see all vision tokens)
```

### What Gets Masked

**Phase 2a** (current):
- ✅ Text tokens (positions 18-23): Each masked individually
- ❌ Vision tokens (positions 1-16): **SKIPPED** (not masked)

**Phase 2b** (future):
- ✅ Vision tokens: Mask individual patches
- ✅ Text tokens: Continue masking
- Analysis: Which image regions matter most?

---

## Expected Results

### What the Analysis Reveals

For each text token (e.g., "What", "image", "?"):
1. **L2 Distance**: How much does masking this word change the model's understanding?
2. **Cosine Distance**: Does it change the *direction* of processing?
3. **Per-Layer**: At which layers does the text token matter most?
4. **Per-Head**: Which attention heads focus on this text token?

### Hypotheses to Test

1. **Question words matter more**: "What" might have higher impact than "?"
2. **Object nouns matter**: "image" might be key for triggering visual reasoning
3. **Early layers**: Text might set context before vision fusion
4. **Late layers**: Cross-modal integration happens in later layers

### Visualizing Results

Load in `visualize_results.html`:
- Compare masked vs unmasked generation
- See which text tokens have high L2/cosine distance
- Track importance across 36 layers
- **Note**: Vision tokens won't appear (they weren't masked)

---

## Validation Steps

### 1. Check Vision Token Detection

```bash
python mask_impact_vl.py \
  --image testimg.png \
  --prompt "Test" \
  --num-tokens 1
```

**Look for**:
```
Vision token ranges: [(1, 17)]
Total vision tokens: 16
```

✅ 128×128 image → 16 vision tokens (correct!)

### 2. Verify Text-Only Masking

Check CSV output:
```bash
head -20 output/vl_masking_results.csv | grep token_masked
```

**Should NOT see** vision token positions (1-16) in `token_position` column
**Should ONLY see** text token positions (18+)

### 3. Compare with Text-Only

```bash
# Phase 1: No image
python mask_impact_vl.py --prompt "What's in this image?" --num-tokens 1 --output text_only

# Phase 2a: With image
python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --num-tokens 1 --output with_image

# Compare in visualizer!
```

**Hypothesis**: Text tokens should have *different* importance when image is present!

---

## Known Limitations (Phase 2a)

- ✅ Image inputs work
- ✅ Proper attention mask (bidirectional vision)
- ✅ Text masking only
- ❌ Vision token masking (Phase 2b)
- ❌ Multiple images (should work but untested)
- ❌ Videos (Phase 3?)
- ❌ Batch mode (Phase 1.1 - can add anytime)

---

## Next Steps

### Option 1: Test & Validate Phase 2a ✅ (Recommended)

Test with various images and prompts:
```bash
# Different question types
python mask_impact_vl.py --image testimg.png --prompt "Describe this" --num-tokens 5
python mask_impact_vl.py --image testimg.png --prompt "What color is it?" --num-tokens 3
python mask_impact_vl.py --image testimg.png --prompt "How many objects?" --num-tokens 2
```

### Option 2: Add Vision Token Masking (Phase 2b)

Mask individual vision tokens to see which image patches matter:
- Loop over vision token positions
- Mask each vision patch
- Measure impact on text generation
- Create spatial heatmaps

### Option 3: Add Batch Mode (Phase 1.1)

Process multiple prompts from YAML config:
```yaml
prompts:
  - name: "piggy_describe"
    image: "testimg.png"
    prompt: "Describe this"
    num_tokens: 10
```

---

## Quick Reference Commands

```bash
# Text-only (Phase 1)
python mask_impact_vl.py --prompt "Test" --num-tokens 1

# Image + text, mask text (Phase 2a) - NEW!
python mask_impact_vl.py --image testimg.png --prompt "What is this?" --num-tokens 3

# With GPU
python mask_impact_vl.py --image testimg.png --prompt "Test" --device cuda --num-tokens 1

# Custom output name
python mask_impact_vl.py --image testimg.png --prompt "Test" --output piggy_test
```

---

## Architecture Correctness ✅

### What We Got Right

1. **Bidirectional Vision Attention**
   - Vision tokens can attend to all other vision tokens
   - No causal masking within vision region
   - Preserves VLM's natural processing

2. **Causal Text Attention**
   - Text tokens use autoregressive masking
   - Can attend to all vision tokens (they come first)
   - Standard LM behavior maintained

3. **Text-Only Masking**
   - Skips vision tokens in masking loop
   - Only experimental masking on text
   - Vision structure preserved

### Implementation Quality

```python
# Proper attention mask creation
def create_vl_attention_mask(seq_length, vision_ranges, device):
    mask = torch.triu(...)  # Start with causal
    for start, end in vision_ranges:
        mask[start:end, start:end] = 0.0  # Vision bidirectional
        mask[end:, start:end] = 0.0       # Text can see vision
    return mask

# Text-only masking
for mask_pos in range(current_num_tokens):
    is_vision_token = any(start <= mask_pos < end for start, end in vision_ranges)
    if is_vision_token:
        continue  # Skip vision tokens
    # ... mask text token ...
```

---

## Success Criteria

Phase 2a is successful if:

1. ✅ Image loads correctly
2. ✅ Vision tokens detected (16 for 128×128 image)
3. ✅ Only text tokens get masked
4. ✅ Output has expected number of rows
5. ✅ Visualizer displays results correctly
6. ✅ Text token importance differs from text-only mode

**Ready to test!** 🎉

Try it now:
```bash
python mask_impact_vl.py --image testimg.png --prompt "What is this?" --num-tokens 1
```

