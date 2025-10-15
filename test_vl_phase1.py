"""
Quick test script for Phase 1: Qwen3-VL text-only masking analysis
Tests basic functionality without full analysis
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def test_model_loading():
    """Test that we can load and access the VL model correctly"""
    print("="*70)
    print("TEST 1: Model Loading and Architecture")
    print("="*70)
    
    model_path = "Qwen/Qwen3-VL-4B-Instruct"
    print(f"\nLoading model: {model_path}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Check architecture
    print(f"✓ Model loaded successfully!")
    print(f"  Text model layers: {len(model.model.language_model.layers)}")
    print(f"  Hidden size: {model.config.text_config.hidden_size}")
    print(f"  Attention heads: {model.config.text_config.num_attention_heads}")
    print(f"  Head dim: {model.config.text_config.head_dim}")
    
    # Check MRoPE config
    if hasattr(model.config.text_config, 'rope_scaling'):
        rope_config = model.config.text_config.rope_scaling
        print(f"  MRoPE config: {rope_config}")
    
    return model, processor


def test_text_processing(model, processor):
    """Test text-only input processing"""
    print("\n" + "="*70)
    print("TEST 2: Text-Only Input Processing")
    print("="*70)
    
    prompt = "What is the capital of France?"
    print(f"\nPrompt: {prompt}")
    
    # Process as text-only
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # Remove token_type_ids if present
    inputs.pop("token_type_ids", None)
    
    print(f"✓ Tokenization successful!")
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Num tokens: {inputs['input_ids'].shape[1]}")
    
    # Decode tokens
    tokens = [processor.tokenizer.decode([tid]) for tid in inputs['input_ids'][0]]
    print(f"  Tokens: {tokens}")
    
    return inputs


def test_forward_pass(model, inputs):
    """Test forward pass through the model"""
    print("\n" + "="*70)
    print("TEST 3: Forward Pass")
    print("="*70)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✓ Forward pass successful!")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Last token logits: {outputs.logits[0, -1, :].shape}")
    
    # Get top-5 predictions
    logits = outputs.logits[0, -1, :]
    top_k_values, top_k_indices = torch.topk(logits, k=5)
    
    print(f"\n  Top-5 predictions for next token:")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    for i, (val, idx) in enumerate(zip(top_k_values, top_k_indices), 1):
        token = processor.tokenizer.decode([idx.item()])
        print(f"    {i}. '{token}' (logit: {val:.2f})")
    
    return outputs


def test_layer_access(model):
    """Test accessing individual layers"""
    print("\n" + "="*70)
    print("TEST 4: Layer Access")
    print("="*70)
    
    num_layers = len(model.model.language_model.layers)
    print(f"\nTotal layers: {num_layers}")
    
    # Check first, middle, and last layer
    test_layers = [0, num_layers // 2, num_layers - 1]
    
    for layer_idx in test_layers:
        layer = model.model.language_model.layers[layer_idx]
        print(f"\n  Layer {layer_idx}:")
        print(f"    Type: {type(layer).__name__}")
        print(f"    Has self_attn: {hasattr(layer, 'self_attn')}")
        print(f"    Has mlp: {hasattr(layer, 'mlp')}")
        print(f"    Has input_layernorm: {hasattr(layer, 'input_layernorm')}")
        
        # Check attention attributes
        attn = layer.self_attn
        print(f"    Attention has q_proj: {hasattr(attn, 'q_proj')}")
        print(f"    Attention has k_proj: {hasattr(attn, 'k_proj')}")
        print(f"    Attention has v_proj: {hasattr(attn, 'v_proj')}")
        print(f"    Attention has o_proj: {hasattr(attn, 'o_proj')}")
    
    print(f"\n✓ All layers accessible!")


def test_rope_embeddings(model, inputs):
    """Test RoPE embedding computation"""
    print("\n" + "="*70)
    print("TEST 5: RoPE Embeddings")
    print("="*70)
    
    input_ids = inputs['input_ids']
    hidden_states = model.model.language_model.embed_tokens(input_ids)
    
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
    
    # Get RoPE embeddings
    position_embeddings = model.model.language_model.rotary_emb(hidden_states, position_ids)
    
    print(f"✓ RoPE embeddings computed!")
    print(f"  Position IDs shape: {position_ids.shape}")
    print(f"  Cos shape: {position_embeddings[0].shape}")
    print(f"  Sin shape: {position_embeddings[1].shape}")
    
    # Check if it's MRoPE or standard RoPE
    if hasattr(model.config.text_config, 'rope_scaling'):
        print(f"  RoPE type: MRoPE (text-only uses first section)")
    else:
        print(f"  RoPE type: Standard")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Qwen3-VL Phase 1 Test Suite")
    print("Testing text-only functionality")
    print("="*70 + "\n")
    
    try:
        # Run tests
        model, processor = test_model_loading()
        inputs = test_text_processing(model, processor)
        outputs = test_forward_pass(model, inputs)
        test_layer_access(model)
        test_rope_embeddings(model, inputs)
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nPhase 1 implementation is ready to use.")
        print("Try: python mask_impact_vl.py --prompt 'Hello world' --num-tokens 1")
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

