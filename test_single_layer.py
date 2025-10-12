"""
Quick test script to run masking analysis on a single layer.
Useful for initial testing and debugging.
"""

import torch
import pandas as pd
from mask_impact_analysis import run_masking_experiment

def test_single_layer():
    """Test with just first 2 layers for speed"""
    prompt = "The Eiffel Tower is in"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Testing on device: {device}")
    print(f"Prompt: '{prompt}'")
    
    # Import after defining the function to allow modification
    from mask_impact_analysis import AttentionMasker
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model
    model_name = r"H:\Models\huggingface\Qwen3-0.6B"
    print(f"Loading model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    
    masker = AttentionMasker(model, tokenizer, device=device)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
    
    print(f"Tokens: {tokens}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    
    # Test just first layer, first token
    print("\n=== Testing Layer 0, masking token 0 ===")
    
    # Baseline
    print("Running baseline...")
    baseline = masker.run_baseline(input_ids, layer_idx=0)
    print(f"Baseline keys: {baseline.keys()}")
    print(f"Baseline resid_pre shape: {baseline['resid_pre'].shape}")
    print(f"Baseline resid_post shape: {baseline['resid_post'].shape}")
    print(f"Baseline attn_output shape: {baseline['attn_output'].shape}")
    print(f"Baseline per_head_outputs shape: {baseline['per_head_outputs'].shape}")
    
    # Masked
    print("\nRunning masked...")
    masked = masker.run_masked(input_ids, layer_idx=0, mask_position=0)
    print(f"Masked keys: {masked.keys()}")
    print(f"Masked resid_pre shape: {masked['resid_pre'].shape}")
    print(f"Masked resid_post shape: {masked['resid_post'].shape}")
    print(f"Masked attn_output shape: {masked['attn_output'].shape}")
    print(f"Masked per_head_outputs shape: {masked['per_head_outputs'].shape}")
    
    # Compute differences
    final_pos = -1
    baseline_full_update = baseline['resid_post'][0, final_pos, :] - baseline['resid_pre'][0, final_pos, :]
    masked_full_update = masked['resid_post'][0, final_pos, :] - masked['resid_pre'][0, final_pos, :]
    
    l2_diff = torch.norm(baseline_full_update - masked_full_update, p=2).item()
    print(f"\nL2 difference in full update: {l2_diff:.6f}")
    
    print("\n[SUCCESS] Test passed! Ready to run full experiment.")

if __name__ == '__main__':
    test_single_layer()

