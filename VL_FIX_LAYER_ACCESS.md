# 🔧 Layer Access Fix for Qwen3-VL

**Issue**: Initial implementation used `model.model.layers` which doesn't exist in Qwen3-VL architecture.

**Root Cause**: The Qwen3-VL model structure is different from standard Qwen3:
- **Qwen3 (text-only)**: `model.model.layers` ✅
- **Qwen3-VL**: `model.model.language_model.layers` ✅

## What Was Fixed

### Files Updated
1. ✅ `mask_impact_vl.py` - All layer/embedding access paths
2. ✅ `test_vl_phase1.py` - Test suite layer access

### Changes Made

| Component | Old Path | New Path |
|-----------|----------|----------|
| **Layers** | `model.model.layers` | `model.model.language_model.layers` |
| **Embeddings** | `model.model.embed_tokens` | `model.model.language_model.embed_tokens` |
| **RoPE** | `model.model.rotary_emb` | `model.model.language_model.rotary_emb` |

## Architecture Insight

```
Qwen3VLForConditionalGeneration
├── model (Qwen3VLModel)
│   ├── visual (Vision encoder - 24 layers)
│   └── language_model (Text decoder - 36 layers) ← WE ACCESS THIS
│       ├── embed_tokens
│       ├── layers[0..35]
│       ├── rotary_emb (MRoPE)
│       └── norm
└── lm_head
```

**Key difference**: VL model wraps the text decoder in `language_model`, whereas pure text models have layers directly under `model`.

## Verification

Run the test suite to verify:
```bash
python test_vl_phase1.py
```

**Expected output**:
```
✓ TEST 1: Model Loading and Architecture
  Text model layers: 36
  Hidden size: 2560
  ...
✓ ALL TESTS PASSED!
```

## Now Try It!

```bash
# Quick test
python mask_impact_vl.py --prompt "Test" --num-tokens 1
```

This should now work correctly! 🎉

---

**Note**: This is a Qwen3-VL architectural quirk. Other VL models (LLaVA, etc.) may have different paths. Always use the debug script to find the correct layer access pattern when adapting to new models.

