"""
Token Masking Analysis: Measure impact of masking individual tokens from attention K/V
on residual stream updates at each layer.
"""

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class MaskingResult:
    """Store results from a single masking experiment"""
    layer: int
    token_masked: str
    token_position: int
    variant: str
    l2_distance: float
    cosine_distance: float
    generation_step: int = 0  # Which output token is being predicted (0 = initial prompt)


class AttentionMasker:
    """Handles masking tokens in attention computation and extracting activations"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Storage for activations
        self.activations = {}
    
    def run_baseline_from_cache(self, cached_hidden_states: torch.Tensor, layer_idx: int,
                                attention_mask: torch.Tensor, position_ids: torch.Tensor,
                                position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run baseline forward pass using pre-computed hidden states.
        This avoids recomputing all layers before layer_idx.
        """
        with torch.no_grad():
            # Use cached hidden states as input to target layer
            hidden_states = cached_hidden_states.clone()
            resid_pre = hidden_states.detach().clone()
            
            # Run target layer normally
            layer = self.model.model.layers[layer_idx]
            
            # Run through the layer manually to capture intermediate activations
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Run attention normally (no masking)
            attn_output, per_head_outputs = self._run_attention(layer, hidden_states, attention_mask, position_ids)
            
            # Store attention output
            attn_output_stored = attn_output.detach().clone()
            
            # Add residual
            hidden_states = residual + attn_output
            
            # MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            # Store output from layer
            resid_post = hidden_states.detach().clone()
            
        return {
            'resid_pre': resid_pre,
            'resid_post': resid_post,
            'attn_output': attn_output_stored,
            'per_head_outputs': per_head_outputs,
        }
    
    def run_baseline(self, input_ids: torch.Tensor, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Run baseline forward pass and extract activations"""
        with torch.no_grad():
            # Get embeddings
            hidden_states = self.model.model.embed_tokens(input_ids)
            
            # Prepare attention mask and position ids
            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Causal mask
            attention_mask = torch.triu(torch.ones((seq_length, seq_length), device=self.device) * float('-inf'), diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            
            # Pre-compute position embeddings for Qwen3
            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            
            # Run through layers before target layer
            for idx in range(layer_idx):
                layer_outputs = self.model.model.layers[idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
            
            # Store input to target layer
            resid_pre = hidden_states.detach().clone()
            
            # Run target layer normally
            layer = self.model.model.layers[layer_idx]
            
            # Run through the layer manually to capture intermediate activations
            attn_input = hidden_states
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Run attention normally (no masking)
            attn_output, per_head_outputs = self._run_attention(layer, hidden_states, attention_mask, position_ids)
            
            # Store attention output
            attn_output_stored = attn_output.detach().clone()
            
            # Add residual
            hidden_states = residual + attn_output
            
            # MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            # Store output from layer
            resid_post = hidden_states.detach().clone()
            
        return {
            'resid_pre': resid_pre,
            'resid_post': resid_post,
            'attn_output': attn_output_stored,
            'per_head_outputs': per_head_outputs,
        }
    
    def run_masked_from_cache(self, cached_hidden_states: torch.Tensor, layer_idx: int,
                             mask_position: int, attention_mask: torch.Tensor,
                             position_ids: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run masked forward pass using pre-computed hidden states.
        This avoids recomputing all layers before layer_idx.
        """
        with torch.no_grad():
            # Use cached hidden states as input to target layer
            hidden_states = cached_hidden_states.clone()
            resid_pre = hidden_states.detach().clone()
            
            # Run target layer with masking
            layer = self.model.model.layers[layer_idx]
            
            # Run through the layer manually to capture intermediate activations
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Custom masked attention
            attn_output, per_head_outputs = self._run_attention(layer, hidden_states, attention_mask, 
                                                                position_ids, mask_position)
            
            # Store attention output
            attn_output_stored = attn_output.detach().clone()
            
            # Add residual
            hidden_states = residual + attn_output
            
            # MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            # Store output from layer
            resid_post = hidden_states.detach().clone()
            
        return {
            'resid_pre': resid_pre,
            'resid_post': resid_post,
            'attn_output': attn_output_stored,
            'per_head_outputs': per_head_outputs,
        }
    
    def run_masked(self, input_ids: torch.Tensor, layer_idx: int, 
                   mask_position: int) -> Dict[str, torch.Tensor]:
        """Run forward pass with masked attention at specific layer"""
        # This is more complex - we need to run through layers manually
        # to inject masking at the target layer
        
        with torch.no_grad():
            # Get embeddings
            hidden_states = self.model.model.embed_tokens(input_ids)
            
            # Prepare attention mask and position ids
            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Causal mask
            attention_mask = torch.triu(torch.ones((seq_length, seq_length), device=self.device) * float('-inf'), diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            
            # Pre-compute position embeddings for Qwen3
            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            
            # Run through layers before target layer
            for idx in range(layer_idx):
                layer_outputs = self.model.model.layers[idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
            
            # Store input to target layer
            resid_pre = hidden_states.detach().clone()
            
            # Run target layer with masking
            layer = self.model.model.layers[layer_idx]
            
            # We need to manually run through the layer to capture intermediate activations
            # 1. Run attention with masking
            attn_input = hidden_states
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Custom masked attention
            attn_output, per_head_outputs = self._run_attention(layer, hidden_states, attention_mask, 
                                                                position_ids, mask_position)
            
            # Store attention output
            attn_output_stored = attn_output.detach().clone()
            
            # Add residual
            hidden_states = residual + attn_output
            
            # MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            # Store output from layer
            resid_post = hidden_states.detach().clone()
            
        return {
            'resid_pre': resid_pre,
            'resid_post': resid_post,
            'attn_output': attn_output_stored,
            'per_head_outputs': per_head_outputs,
        }
    
    def _run_attention(self, layer, hidden_states, attention_mask, position_ids, mask_position=None):
        """
        Helper to run attention with optional masking.
        
        Args:
            layer: The transformer layer
            hidden_states: Input hidden states
            attention_mask: Causal attention mask
            position_ids: Position IDs for rotary embeddings
            mask_position: Optional position to mask (set attention to -inf)
            
        Returns:
            (attn_output, per_head_outputs): Combined attention output and per-head outputs
        """
        # Get Q, K, V projections
        bsz, q_len, _ = hidden_states.size()
        
        query_states = layer.self_attn.q_proj(hidden_states)
        key_states = layer.self_attn.k_proj(hidden_states)
        value_states = layer.self_attn.v_proj(hidden_states)
        
        # Get attention parameters (handle different naming conventions)
        num_heads = getattr(layer.self_attn, 'num_heads', None) or getattr(layer.self_attn, 'num_attention_heads', self.model.config.num_attention_heads)
        num_key_value_heads = getattr(layer.self_attn, 'num_key_value_heads', self.model.config.num_key_value_heads)
        head_dim = getattr(layer.self_attn, 'head_dim', self.model.config.head_dim)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        
        # Apply query/key normalization if present (Qwen3)
        if hasattr(layer.self_attn, 'q_norm'):
            query_states = layer.self_attn.q_norm(query_states)
        if hasattr(layer.self_attn, 'k_norm'):
            key_states = layer.self_attn.k_norm(key_states)
        
        # Apply rotary embeddings (get from model level for Qwen3)
        rotary_emb = getattr(layer.self_attn, 'rotary_emb', self.model.model.rotary_emb)
        cos, sin = rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Repeat KV heads if using GQA
        if num_key_value_heads != num_heads:
            key_states = repeat_kv(key_states, num_heads // num_key_value_heads)
            value_states = repeat_kv(value_states, num_heads // num_key_value_heads)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(head_dim)
        
        # Apply masking if specified: set attention logits to -inf for masked position
        if mask_position is not None:
            attn_weights[:, :, :, mask_position] = float('-inf')
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Store per-head outputs before combining
        per_head_outputs = attn_output.detach().clone()  # [batch, num_heads, seq_len, head_dim]
        
        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
        
        # Output projection
        attn_output = layer.self_attn.o_proj(attn_output)
        
        return attn_output, per_head_outputs


def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads for grouped-query attention"""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embeddings"""
    # Handle different shapes
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_distances(baseline: torch.Tensor, masked: torch.Tensor) -> Tuple[float, float]:
    """
    Compute L2 and cosine distances between baseline and masked activations.
    
    Args:
        baseline: Baseline activation tensor
        masked: Masked activation tensor
        
    Returns:
        (l2_distance, cosine_distance)
    """
    # Flatten tensors
    baseline_flat = baseline.flatten()
    masked_flat = masked.flatten()
    
    # L2 distance
    l2_dist = torch.norm(baseline_flat - masked_flat, p=2).item()
    
    # Cosine distance
    cosine_sim = F.cosine_similarity(baseline_flat.unsqueeze(0), masked_flat.unsqueeze(0), dim=-1).item()
    cosine_dist = 1 - cosine_sim
    
    return l2_dist, cosine_dist


def run_masking_experiment(prompt: str, model_name: str = r"H:\Models\huggingface\hub\models--Qwen--Qwen3-4B-Instruct-2507\snapshots\cdbee75f17c01a7cc42f958dc650907174af0554",
                          device: str = 'cpu', num_output_tokens: int = 1) -> pd.DataFrame:
    """
    Run the complete masking experiment.
    
    Args:
        prompt: Input prompt to analyze
        model_name: HuggingFace model name
        device: 'cpu' or 'cuda'
        num_output_tokens: Number of tokens to generate for baseline prediction (default: 1)
        
    Returns:
        DataFrame with results in long format
    """
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    
    masker = AttentionMasker(model, tokenizer, device=device)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    
    # Decode tokens for reporting
    tokens = [tokenizer.decode([token_id]) for token_id in input_ids[0]]
    num_tokens = len(tokens)
    num_layers = model.config.num_hidden_layers
    
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {tokens}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of tokens: {num_tokens}")
    
    results = []
    
    # For each generation step (analyzing progressively longer sequences)
    current_input_ids = input_ids.clone()
    
    # Get EOS token ID(s) for early stopping
    eos_token_id = tokenizer.eos_token_id
    im_end_token = "<|im_end|>"
    im_end_token_id = tokenizer.encode(im_end_token, add_special_tokens=False)[-1] if im_end_token in tokenizer.get_vocab() else None
    
    for gen_step in range(num_output_tokens):
        if gen_step > 0:
            # Generate next token
            with torch.no_grad():
                outputs = masker.model(current_input_ids)
                logits = outputs.logits[0, -1, :]
                predicted_token_id = torch.argmax(logits).item()
                predicted_token = tokenizer.decode([predicted_token_id])
                
                # Append to input
                current_input_ids = torch.cat([current_input_ids, torch.tensor([[predicted_token_id]], device=device)], dim=1)
                
                print(f"\n{'='*70}")
                print(f"Generation step {gen_step}: Generated token '{predicted_token}'")
                
                # Check for EOS or <|im_end|> token
                if predicted_token_id == eos_token_id or predicted_token_id == im_end_token_id:
                    print(f"Stopping early: Generated end token ('{predicted_token}')")
                    print(f"{'='*70}\n")
                    break
                
                print(f"{'='*70}\n")
        
        # Update tokens and num_tokens for current input
        current_tokens = [tokenizer.decode([token_id]) for token_id in current_input_ids[0]]
        current_num_tokens = len(current_tokens)
        
        if gen_step == 0:
            print(f"\nAnalyzing initial prompt ({current_num_tokens} tokens)...")
        else:
            print(f"Analyzing with {current_num_tokens} tokens (original + {gen_step} generated)...")
        
        # PRE-COMPUTE: Run forward pass once to get hidden states at each layer input
        # This avoids recomputing layers 0 to L-1 for every experiment at layer L
        print(f"Pre-computing hidden states for all {num_layers} layers...")
        cached_hidden_states = {}  # layer_idx -> hidden_states at input to that layer
        
        with torch.no_grad():
            hidden_states = masker.model.model.embed_tokens(current_input_ids)
            batch_size, seq_length = current_input_ids.shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.triu(torch.ones((seq_length, seq_length), device=device) * float('-inf'), diagonal=1)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            position_embeddings = masker.model.model.rotary_emb(hidden_states, position_ids)
            
            # Cache initial embeddings as input to layer 0
            cached_hidden_states[0] = hidden_states.detach().clone()
            
            # Run through each layer and cache the output as input to next layer
            for idx in range(num_layers):
                if idx % 5 == 0 or idx == num_layers - 1:  # Print every 5 layers
                    print(f"  Pre-computing layer {idx}/{num_layers-1}...")
                layer_outputs = masker.model.model.layers[idx](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
                # Cache for next layer (if not the last layer)
                if idx < num_layers - 1:
                    cached_hidden_states[idx + 1] = hidden_states.detach().clone()
        
        print("Pre-computation complete! Starting layer-by-layer analysis...")
        
        # For each layer
        for layer_idx in range(num_layers):
            print(f"  Layer {layer_idx}/{num_layers-1}...")
            
            # Use cached hidden states instead of recomputing layers 0 to L-1
            baseline_activations = masker.run_baseline_from_cache(
                cached_hidden_states[layer_idx], layer_idx, attention_mask, position_ids, position_embeddings
            )
            
            # Extract activations at final token position
            final_pos = -1
            baseline_resid_pre = baseline_activations['resid_pre'][0, final_pos, :]
            baseline_resid_post = baseline_activations['resid_post'][0, final_pos, :]
            baseline_attn_output = baseline_activations['attn_output'][0, final_pos, :]
            
            # Compute baseline updates
            baseline_full_update = baseline_resid_post - baseline_resid_pre
            baseline_attn_update = baseline_attn_output  # Attention contribution
            
            # For each token to mask (mask all tokens in current sequence)
            for mask_pos in range(current_num_tokens):
                token_str = current_tokens[mask_pos]
                
                # Run masked forward pass using cached hidden states
                masked_activations = masker.run_masked_from_cache(
                    cached_hidden_states[layer_idx], layer_idx, mask_pos, 
                    attention_mask, position_ids, position_embeddings
                )
                
                # Extract activations at final token position
                masked_resid_pre = masked_activations['resid_pre'][0, final_pos, :]
                masked_resid_post = masked_activations['resid_post'][0, final_pos, :]
                masked_attn_output = masked_activations['attn_output'][0, final_pos, :]
                
                # Compute masked updates
                masked_full_update = masked_resid_post - masked_resid_pre
                masked_attn_update = masked_attn_output
                
                # Variant 1: Full layer update
                l2_full, cos_full = compute_distances(baseline_full_update, masked_full_update)
                results.append(MaskingResult(
                    layer=layer_idx,
                    token_masked=token_str,
                    token_position=mask_pos,
                    variant='Full',
                    l2_distance=l2_full,
                    cosine_distance=cos_full,
                    generation_step=gen_step
                ))
                
                # Variant 2: Attention-only update
                l2_attn, cos_attn = compute_distances(baseline_attn_update, masked_attn_update)
                results.append(MaskingResult(
                    layer=layer_idx,
                    token_masked=token_str,
                    token_position=mask_pos,
                    variant='Attn',
                    l2_distance=l2_attn,
                    cosine_distance=cos_attn,
                    generation_step=gen_step
                ))
                
                # Variant 3: Per-head updates
                if 'per_head_outputs' in masked_activations and 'per_head_outputs' in baseline_activations:
                    baseline_heads = baseline_activations['per_head_outputs']
                    masked_heads = masked_activations['per_head_outputs']
                    
                    # Shape: [batch, num_heads, seq_len, head_dim]
                    num_heads = masked_heads.shape[1]
                    for head_idx in range(num_heads):
                        # Extract head output at final position for this specific head
                        baseline_head = baseline_heads[0, head_idx, final_pos, :]  # [head_dim]
                        masked_head = masked_heads[0, head_idx, final_pos, :]  # [head_dim]
                        
                        # Compute distances
                        l2_head, cos_head = compute_distances(baseline_head, masked_head)
                        results.append(MaskingResult(
                            layer=layer_idx,
                            token_masked=token_str,
                            token_position=mask_pos,
                            variant=f'Head_{head_idx}',
                            l2_distance=l2_head,
                            cosine_distance=cos_head,
                            generation_step=gen_step
                        ))
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'generation_step': r.generation_step,
            'layer': r.layer,
            'token_masked': r.token_masked,
            'token_position': r.token_position,
            'variant': r.variant,
            'l2_distance': r.l2_distance,
            'cosine_distance': r.cosine_distance
        }
        for r in results
    ])
    
    # Generate prediction without masking (temp=0, greedy sampling)
    print("\n" + "="*70)
    print(f"Baseline Prediction (no masking, temperature=0, {num_output_tokens} token{'s' if num_output_tokens > 1 else ''}):")
    print("="*70)
    with torch.no_grad():
        # Generate multiple tokens autoregressively
        generated_ids = input_ids.clone()
        generated_tokens = []
        early_stop = False
        
        for step in range(num_output_tokens):
            outputs = masker.model(generated_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for final position
            predicted_token_id = torch.argmax(logits).item()
            predicted_token = tokenizer.decode([predicted_token_id])
            generated_tokens.append(predicted_token)
            
            # Append predicted token for next iteration
            generated_ids = torch.cat([generated_ids, torch.tensor([[predicted_token_id]], device=device)], dim=1)
            
            # Show top-5 for first token
            if step == 0:
                top_k_values, top_k_indices = torch.topk(logits, k=5)
                top_k_tokens = [tokenizer.decode([idx.item()]) for idx in top_k_indices]
                top_k_probs = torch.softmax(top_k_values, dim=0)
            
            # Check for EOS or <|im_end|> token
            if predicted_token_id == eos_token_id or predicted_token_id == im_end_token_id:
                early_stop = True
                break
        
        print(f"Input prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        if num_output_tokens == 1:
            print(f"\nGreedy prediction (argmax): '{generated_tokens[0]}'")
            print(f"\nTop 5 predictions for next token:")
            for i, (token, prob) in enumerate(zip(top_k_tokens, top_k_probs), 1):
                print(f"  {i}. '{token}' (prob: {prob:.4f})")
        else:
            full_generation = ''.join(generated_tokens)
            actual_tokens = len(generated_tokens)
            stop_msg = f" (stopped early at {actual_tokens} tokens)" if early_stop else ""
            print(f"\nGenerated sequence ({actual_tokens}/{num_output_tokens} tokens{stop_msg}):")
            print(f"  {full_generation}")
            print(f"\nToken-by-token:")
            for i, token in enumerate(generated_tokens, 1):
                print(f"  {i}. '{token}'")
            if early_stop:
                print(f"\n⚠ Generation stopped early: Model generated end token")
            if num_output_tokens > 0:
                print(f"\nTop 5 predictions for first token:")
                for i, (token, prob) in enumerate(zip(top_k_tokens, top_k_probs), 1):
                    print(f"  {i}. '{token}' (prob: {prob:.4f})")
    print("="*70 + "\n")
    
    return df


if __name__ == '__main__':
    import sys
    import argparse
    import yaml
    import os
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description='Analyze token masking impact on transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt from prompt.txt (legacy mode)
  python mask_impact_analysis.py --prompt prompt.txt --num-tokens 5
  
  # Batch mode with YAML config
  python mask_impact_analysis.py --config prompts_config.yaml
  
  # Override device
  python mask_impact_analysis.py --config prompts_config.yaml --device cuda
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to YAML config file for batch processing')
    parser.add_argument('--prompt', type=str, help='Single prompt text or path to .txt file')
    parser.add_argument('--num-tokens', type=int, default=1, help='Number of tokens to generate (single prompt mode)')
    parser.add_argument('--output', type=str, default='masking_results', help='Output file basename (single prompt mode)')
    parser.add_argument('--model', type=str, help='Override model path')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # BATCH MODE: Process YAML config with multiple prompts
    if args.config:
        print(f"\n{'='*70}")
        print(f"BATCH MODE: Loading config from {args.config}")
        print(f"{'='*70}\n")
        
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Config file '{args.config}' not found")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            sys.exit(1)
        
        # Get model path (command line overrides config)
        model_name = args.model if args.model else config.get('model', {}).get('path')
        
        prompts = config.get('prompts', [])
        if not prompts:
            print("Error: No prompts found in config file")
            sys.exit(1)
        
        # Filter to only enabled prompts
        enabled_prompts = [p for p in prompts if p.get('enabled', True)]
        disabled_count = len(prompts) - len(enabled_prompts)
        
        if disabled_count > 0:
            print(f"Found {len(prompts)} prompt(s) total ({len(enabled_prompts)} enabled, {disabled_count} disabled)\n")
        else:
            print(f"Found {len(enabled_prompts)} prompt(s) to process\n")
        
        if not enabled_prompts:
            print("Error: All prompts are disabled")
            sys.exit(1)
        
        for idx, prompt_config in enumerate(enabled_prompts, 1):
            name = prompt_config.get('name', f'prompt_{idx}')
            prompt_text = prompt_config.get('prompt', '')
            num_tokens = prompt_config.get('num_tokens', 1)
            
            # If prompt is a .txt file path, load it
            if prompt_text.endswith('.txt') and os.path.exists(prompt_text):
                try:
                    with open(prompt_text, 'r', encoding='utf-8') as f:
                        prompt_text = f.read()
                    print(f"[{idx}/{len(enabled_prompts)}] Processing '{name}' (loaded from {prompt_config.get('prompt')})")
                except Exception as e:
                    print(f"Error loading {prompt_text}: {e}")
                    continue
            else:
                print(f"[{idx}/{len(enabled_prompts)}] Processing '{name}'")
            
            print(f"  Prompt length: {len(prompt_text)} chars")
            print(f"  Generating: {num_tokens} token(s)")
            
            # Run experiment
            df = run_masking_experiment(
                prompt_text, 
                model_name=model_name,
                device=device, 
                num_output_tokens=num_tokens
            )
            
            # Save with custom name in output directory
            csv_file = output_dir / f'{name}_results.csv'
            json_file = output_dir / f'{name}_results.json'
            
            df.to_csv(csv_file, index=False)
            df.to_json(json_file, orient='records')
            
            print(f"  ✓ Saved to: {csv_file} and {json_file}")
            print(f"  Rows: {len(df)}\n")
        
        print(f"{'='*70}")
        print(f"Batch processing complete! Processed {len(enabled_prompts)} prompt(s)")
        if disabled_count > 0:
            print(f"Skipped {disabled_count} disabled prompt(s)")
        print(f"{'='*70}")
    
    # SINGLE PROMPT MODE: Legacy behavior
    else:
        print(f"\n{'='*70}")
        print(f"SINGLE PROMPT MODE")
        print(f"{'='*70}\n")
        
        # Get prompt from command line or prompt.txt
        if args.prompt:
            if args.prompt.endswith('.txt') and os.path.exists(args.prompt):
                with open(args.prompt, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                print(f"Loaded prompt from {args.prompt}")
            else:
                prompt = args.prompt
                print(f"Using prompt from command line")
        else:
            # Default: try prompt.txt
            try:
                with open('prompt.txt', 'r', encoding='utf-8') as f:
                    prompt = f.read()
                print(f"Loaded prompt from prompt.txt")
            except FileNotFoundError:
                print(f"Warning: prompt.txt not found, using default prompt")
                prompt = "The Eiffel Tower is in"
        
        # Get model path
        model_name = args.model if args.model else None
        
        # Run experiment
        df = run_masking_experiment(
            prompt, 
            model_name=model_name if model_name else r"H:\Models\huggingface\hub\models--Qwen--Qwen3-4B-Instruct-2507\snapshots\cdbee75f17c01a7cc42f958dc650907174af0554",
            device=device, 
            num_output_tokens=args.num_tokens
        )
        
        # Save to output files in output directory
        csv_file = output_dir / f'{args.output}.csv'
        json_file = output_dir / f'{args.output}.json'
        
        df.to_csv(csv_file, index=False)
        df.to_json(json_file, orient='records')
        
        print(f"\nResults saved to:")
        print(f"  - {csv_file} (for spreadsheet analysis)")
        print(f"  - {json_file} (for fast web visualization)")
        print(f"\nFirst few rows:")
        print(df.head(20))

