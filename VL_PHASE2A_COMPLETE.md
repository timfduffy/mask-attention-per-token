# ✅ Phase 2a Complete: Image + Text Masking!

**Implementation Date**: Just completed!
**Status**: ✅ Working and tested
**What's New**: Analyze text tokens with image inputs

---

## 🎉 What Phase 2a Brings

### Core Features

1. **Image Input Support**
   - Load images via `--image` flag
   - PIL image processing
   - Automatic vision token detection
   - Works with any image size

2. **Proper VL Attention Architecture**
   - Vision tokens: Bidirectional attention (can attend to each other)
   - Text tokens: Causal attention (autoregressive)
   - Text can attend to all vision tokens
   - Preserves natural VLM processing

3. **Text-Only Masking**
   - Masks only text token positions
   - Skips vision tokens automatically
   - Measures: "Which text matters when image is present?"
   - Enables cross-modal analysis

---

## 🚀 How to Use

### Basic Usage

```bash
# Phase 2a: Analyze text with image
python mask_impact_vl.py \
  --image testimg.png \
  --prompt "What's in this image?" \
  --num-tokens 3
```

**Expected Output**:
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

Analyzing initial prompt...
  (skipping 16 vision tokens - masking text only)
...
```

### Compare With and Without Image

```bash
# Text-only (Phase 1)
python mask_impact_vl.py \
  --prompt "What's in this image?" \
  --num-tokens 5 \
  --output text_only

# Image + text (Phase 2a)
python mask_impact_vl.py \
  --image testimg.png \
  --prompt "What's in this image?" \
  --num-tokens 5 \
  --output with_image

# Load both in visualize_results.html to compare!
```

### With GPU

```bash
python mask_impact_vl.py \
  --image photo.jpg \
  --prompt "Describe this" \
  --device cuda \
  --num-tokens 10
```

---

## 📊 What You Can Learn

### Research Questions Answered

1. **Text Context Importance**
   - Which text tokens matter most when image is present?
   - Do question words ("What", "Describe") have higher impact?
   - Does the image change text token importance?

2. **Cross-Modal Dependencies**
   - How does text prompt guide vision processing?
   - Which layers integrate vision and text?
   - Are early or late layers more affected?

3. **Per-Layer Analysis**
   - Track text importance across all 36 layers
   - See where cross-modal fusion happens
   - Identify critical integration points

4. **Per-Head Analysis**
   - Which of 32 attention heads focus on text?
   - Do certain heads specialize in cross-modal attention?
   - Head-level importance patterns

### Example Hypotheses

- **H1**: Question words ("What", "How", "Describe") will show higher L2 distance when masked
- **H2**: Text token importance will differ from text-only mode
- **H3**: Later layers (20+) will show stronger text-vision integration
- **H4**: Specific attention heads specialize in cross-modal processing

---

## 🔬 Technical Details

### Token Sequence Structure

```
Input: image + "What's in this image?"

Token sequence:
[<|vision_start|>] [vision_token_1] ... [vision_token_16] [<|vision_end|>] [What] ['s] [in] [this] [image] [?]
   Position 0           1-16 (BIDIRECTIONAL)                  Position 17      18-23 (CAUSAL)
```

### Attention Mask Matrix

```
                vision tokens (1-16)    text tokens (18-23)
vision (1-16)   [   BIDIRECTIONAL   ]   [     0.0      ]
text (18-23)    [       0.0         ]   [   CAUSAL     ]

Legend:
- BIDIRECTIONAL: All positions can attend to each other (0.0)
- CAUSAL: Upper triangle is -inf (can only attend to earlier positions)
- 0.0: Can attend (text can see all vision tokens)
```

### Vision Token Detection

```python
# Automatic detection via special tokens
vision_start_id = processor.tokenizer.encode("<|vision_start|>", ...)[-1]
vision_end_id = processor.tokenizer.encode("<|vision_end|>", ...)[-1]

# Find ranges
vision_ranges = []  # e.g., [(1, 17)] for single 128×128 image
for i, token_id in enumerate(input_ids[0]):
    if token_id == vision_start_id:
        start_idx = i + 1
    elif token_id == vision_end_id:
        vision_ranges.append((start_idx, i))
```

### Text-Only Masking Logic

```python
# Only mask text tokens, skip vision
for mask_pos in range(current_num_tokens):
    is_vision_token = any(start <= mask_pos < end for start, end in vision_ranges)
    if is_vision_token:
        continue  # Skip vision tokens
    
    # Mask this text token and measure impact
    masked_activations = masker.run_masked_from_cache(...)
```

---

## 📈 Image Size → Vision Tokens

| Image Size | Base Patches (16×16) | After Merge (32×32) | Vision Tokens |
|------------|---------------------|---------------------|---------------|
| 128×128    | 8×8 = 64           | 4×4                | 16            |
| 256×256    | 16×16 = 256        | 8×8                | 64            |
| 512×512    | 32×32 = 1024       | 16×16              | 256           |
| 1024×1024  | 64×64 = 4096       | 32×32              | 1024          |

**Note**: Larger images = more vision tokens = longer analysis time

---

## ✅ Validation Checklist

- [x] Image loads correctly
- [x] Vision token ranges detected
- [x] Vision tokens skipped in masking
- [x] Text tokens masked individually
- [x] Proper attention mask (bidirectional vision, causal text)
- [x] Output has expected number of rows
- [x] CSV/JSON files generated
- [x] Visualizer displays results
- [x] Multi-token generation works with images

---

## 📁 Output Files

```
output/
├── vl_masking_results.csv          # Default output (CSV)
├── vl_masking_results.json         # Default output (JSON)
├── with_image.csv                  # Custom output
└── with_image.json                 # Custom output
```

**Load in visualizer**: `visualize_results.html` → Load JSON file

---

## 🆚 Phase Comparison

| Feature | Phase 1 | Phase 2a | Phase 2b (Next) |
|---------|---------|----------|-----------------|
| **Text-only input** | ✅ | ✅ | ✅ |
| **Image input** | ❌ | ✅ | ✅ |
| **Mask text tokens** | ✅ | ✅ | ✅ |
| **Mask vision tokens** | N/A | ❌ | ✅ |
| **Proper VL attention** | N/A | ✅ | ✅ |
| **Cross-modal analysis** | N/A | ✅ (text→vision) | ✅ (both) |
| **Spatial heatmaps** | N/A | ❌ | ✅ |

---

## 🎯 What's Different from Phase 1?

### Phase 1 (Text-only)
```bash
python mask_impact_vl.py --prompt "What is AI?" --num-tokens 5
```
- No images
- All tokens masked
- Standard causal attention
- Text-only analysis

### Phase 2a (Image + Text)
```bash
python mask_impact_vl.py --image test.png --prompt "What is this?" --num-tokens 5
```
- Images supported! 🎨
- Only text tokens masked
- Bidirectional vision attention
- Cross-modal analysis enabled

---

## 🔍 Example Use Cases

### 1. Analyze Question Types

```bash
# Descriptive
python mask_impact_vl.py --image photo.jpg --prompt "Describe this image" --output desc

# Interrogative
python mask_impact_vl.py --image photo.jpg --prompt "What is in this image?" --output what

# Analytical
python mask_impact_vl.py --image photo.jpg --prompt "How many objects are there?" --output count
```

**Compare**: Which question format produces different text token importance?

### 2. Context Sensitivity

```bash
# No context
python mask_impact_vl.py --image dog.jpg --prompt "What is this?" --output no_context

# With context
python mask_impact_vl.py --image dog.jpg --prompt "What breed of dog is this?" --output with_context
```

**Compare**: Does additional context change which words matter?

### 3. Multi-Token Generation

```bash
python mask_impact_vl.py \
  --image scene.jpg \
  --prompt "Describe this scene" \
  --num-tokens 20 \
  --device cuda
```

**Analyze**: How does text importance evolve as generation progresses?

---

## 🚧 Current Limitations (Phase 2a)

- ❌ **No vision token masking**: Can't analyze which image regions matter
- ❌ **No batch mode**: Single prompts only (can add in Phase 2.1)
- ❌ **No spatial heatmaps**: Can't visualize important image patches
- ❌ **No per-patch analysis**: Vision tokens treated as a block

**These will be addressed in Phase 2b!**

---

## ⏭️ What's Next?

### Option A: Test Phase 2a Thoroughly

Experiment with different:
- Image types (photos, drawings, diagrams)
- Question formats (What/How/Describe/Count)
- Prompt lengths
- Image sizes
- Multi-token generation

### Option B: Implement Phase 2b (Vision Token Masking)

Add ability to mask individual vision tokens:
- Measure which image patches matter most
- Create spatial heatmaps
- Analyze per-patch importance
- Link patches to generated text

### Option C: Add Batch Mode (Phase 2.1)

Process multiple image+text pairs from YAML config:
```yaml
prompts:
  - name: "piggy_describe"
    image: "testimg.png"
    prompt: "Describe this"
    num_tokens: 10
```

---

## 💡 Tips for Analysis

1. **Start Small**: Use `--num-tokens 1` for quick tests
2. **Use GPU**: Phase 2a is ~2x slower than Phase 1, GPU helps
3. **Compare Modes**: Run same prompt with/without image
4. **Track Patterns**: Look for which text positions consistently matter
5. **Layer Analysis**: Check where cross-modal fusion happens (likely layers 15-30)
6. **Head Analysis**: Identify heads specializing in text-vision interaction

---

## 📚 Documentation

- **`VL_IMPLEMENTATION.md`**: Technical roadmap (updated with Phase 2a)
- **`VL_PHASE2A_READY.md`**: Testing guide for Phase 2a
- **`IMPLEMENTATION_COMPLETE.md`**: Master summary (updated)
- **`README.md`**: Project overview (updated)

---

## 🎉 Success!

Phase 2a is **complete and working**! You can now:

✅ Load images
✅ Analyze text tokens in multimodal context
✅ Compare with text-only analysis
✅ Track cross-modal attention patterns
✅ Generate visualizations

**Test it now**:
```bash
python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --num-tokens 3
```

Enjoy exploring cross-modal transformer attention! 🚀🎨

