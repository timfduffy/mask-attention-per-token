"""
Token Masking Analysis for Vision-Language Models (Qwen3-VL)
Phase 1: Text-only inputs to validate architecture compatibility

Measure impact of masking individual tokens from attention K/V on residual stream updates.
"""

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
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


class VLAttentionMasker:
    """Handles masking tokens in VL model attention computation and extracting activations"""
    
    def __init__(self, model, processor, device='cpu'):
        self.model = model.to(device)
        self.processor = processor
        self.tokenizer = processor.tokenizer
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
            
            # Run target layer normally - access text model layers via language_model
            layer = self.model.model.language_model.layers[layer_idx]
            
            # Run through the layer manually to capture intermediate activations
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Run attention normally (no masking)
            attn_output, per_head_outputs = self._run_attention(
                layer, hidden_states, attention_mask, position_ids, position_embeddings
            )
            
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
                             attention_mask: torch.Tensor, position_ids: torch.Tensor,
                             position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                             mask_position: int) -> Dict[str, torch.Tensor]:
        """
        Run masked forward pass using pre-computed hidden states.
        Masks a specific token position from attention.
        """
        with torch.no_grad():
            # Use cached hidden states as input to target layer
            hidden_states = cached_hidden_states.clone()
            resid_pre = hidden_states.detach().clone()
            
            # Get target layer - access text model layers via language_model
            layer = self.model.model.language_model.layers[layer_idx]
            
            # Run through the layer with masking
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Run masked attention
            attn_output, per_head_outputs = self._run_attention(
                layer, hidden_states, attention_mask, position_ids,
                position_embeddings, mask_position=mask_position
            )
            
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
    
    def _run_attention(self, layer, hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                      position_ids: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                      mask_position: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run attention computation with optional token masking.
        Adapted for Qwen3-VL architecture with MRoPE.
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Get attention module
        attn = layer.self_attn
        
        # Get number of heads from config (Qwen3-VL specific)
        num_heads = self.model.config.text_config.num_attention_heads
        num_key_value_heads = self.model.config.text_config.num_key_value_heads
        head_dim = self.model.config.text_config.head_dim
        
        # Project Q, K, V
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
        
        # Apply RoPE (Qwen3-VL uses MRoPE, but text-only uses standard section)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Expand key/value states for GQA if needed
        if num_key_value_heads != num_heads:
            key_states = repeat_kv(key_states, num_heads // num_key_value_heads)
            value_states = repeat_kv(value_states, num_heads // num_key_value_heads)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(head_dim)
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply token masking if specified (set specific position to -inf)
        if mask_position is not None:
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights[:, :, :, mask_position] = mask_value
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout (but we're in eval mode so this is a no-op)
        attn_weights = F.dropout(attn_weights, p=0.0, training=False)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Store per-head outputs before combining
        per_head_outputs = attn_output.detach().clone()  # [bsz, num_heads, q_len, head_dim]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
        attn_output = attn.o_proj(attn_output)
        
        return attn_output, per_head_outputs


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    # Qwen3 uses standard RoPE application
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    """Repeat key/value tensors for Grouped Query Attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def compute_distances(baseline: torch.Tensor, masked: torch.Tensor) -> Tuple[float, float]:
    """Compute L2 and cosine distances between baseline and masked vectors"""
    # L2 distance
    l2_dist = torch.norm(baseline - masked).item()
    
    # Cosine distance (1 - cosine similarity)
    cos_sim = F.cosine_similarity(baseline, masked, dim=0).item()
    cos_dist = 1 - cos_sim
    
    return l2_dist, cos_dist


def create_vl_attention_mask(seq_length: int, vision_ranges: List[Tuple[int, int]], device: str) -> torch.Tensor:
    """
    Create proper attention mask for VL models:
    - Vision tokens: bidirectional attention (can attend to each other)
    - Text tokens: causal attention (autoregressive)
    - Text can attend to all vision tokens (vision comes first)
    
    Args:
        seq_length: Total sequence length
        vision_ranges: List of (start, end) tuples for vision token positions
        device: Device for tensor
    
    Returns:
        Attention mask of shape [1, 1, seq_length, seq_length]
    """
    # Start with causal mask (all text tokens are causal)
    mask = torch.triu(torch.ones((seq_length, seq_length), device=device) * float('-inf'), diagonal=1)
    
    # For each vision token range, make it bidirectional
    for start, end in vision_ranges:
        # Vision tokens can attend to each other bidirectionally
        mask[start:end, start:end] = 0.0
        
        # All later tokens (text) can attend to vision tokens
        mask[end:, start:end] = 0.0
    
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def run_vl_masking_experiment(
    prompt: str,
    model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    device: str = 'cpu',
    num_output_tokens: int = 1,
    image_path: Optional[str] = None,
    mask_mode: str = 'text'
) -> pd.DataFrame:
    """
    Run the complete VL masking experiment
    
    Args:
        prompt: Text prompt to analyze
        model_path: Path to Qwen3-VL model
        device: 'cpu' or 'cuda'
        num_output_tokens: Number of tokens to generate and analyze
        image_path: Optional path to image file (Phase 2)
        mask_mode: 'text', 'vision', or 'both' - which tokens to mask
    """
    # Determine phase based on inputs
    if not image_path:
        phase = "PHASE 1: Text-Only"
    elif mask_mode == 'text':
        phase = "PHASE 2a: Image + Text (masking text only)"
    elif mask_mode == 'vision':
        phase = "PHASE 2b: Image + Text (masking vision only)"
    else:
        phase = "PHASE 2c: Image + Text (masking both text and vision)"
    
    print(f"\n{'='*70}")
    print(f"{phase} Masking Analysis")
    print(f"{'='*70}\n")
    
    # Load model and processor
    print(f"Loading Qwen3-VL model from: {model_path}")
    print(f"Loading model on {device}...")
    
    # Simple loading like the text-only version - just load and move to device
    # Using float32 for CPU ensures full compatibility and proper RAM loading
    if device == 'cpu':
        # CPU: Use float32 for better compatibility and ensure full loading
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
    else:
        # GPU: Can use bfloat16 for speed
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
    
    model = model.to(device)
    print(f"✓ Model loaded successfully on {device}")
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f"Model loaded successfully!")
    print(f"Text model layers: {len(model.model.language_model.layers)}")
    print(f"Hidden size: {model.config.text_config.hidden_size}")
    print(f"Attention heads: {model.config.text_config.num_attention_heads}")
    
    # Process input (with or without image)
    content = []
    if image_path:
        from PIL import Image
        print(f"\nLoading image: {image_path}")
        image = Image.open(image_path)
        print(f"  Image size: {image.size}")
        content.append({"type": "image", "image": image})
    
    content.append({"type": "text", "text": prompt})
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Tokenize using processor
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)
    
    # Remove token_type_ids if present (not used by model)
    inputs.pop("token_type_ids", None)
    
    input_ids = inputs['input_ids']
    
    # Decode tokens for display
    tokens = [processor.tokenizer.decode([token_id]) for token_id in input_ids[0]]
    
    # Detect vision tokens (Phase 2)
    vision_start_id = processor.tokenizer.encode("<|vision_start|>", add_special_tokens=False)[-1] if "<|vision_start|>" in processor.tokenizer.get_vocab() else model.config.vision_start_token_id
    vision_end_id = processor.tokenizer.encode("<|vision_end|>", add_special_tokens=False)[-1] if "<|vision_end|>" in processor.tokenizer.get_vocab() else model.config.vision_end_token_id
    
    # Find vision token ranges
    vision_ranges = []
    if image_path:
        in_vision = False
        start_idx = None
        for i, token_id in enumerate(input_ids[0]):
            if token_id.item() == vision_start_id:
                start_idx = i + 1
                in_vision = True
            elif token_id.item() == vision_end_id and in_vision:
                vision_ranges.append((start_idx, i))
                in_vision = False
        
        print(f"  Vision token ranges: {vision_ranges}")
        if vision_ranges:
            total_vision_tokens = sum(end - start for start, end in vision_ranges)
            print(f"  Total vision tokens: {total_vision_tokens}")
    
    print(f"\nPrompt: {prompt}")
    print(f"Tokenized ({len(tokens)} tokens): {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    if image_path:
        print(f"  (includes {len(vision_ranges)} image(s) with vision tokens)")
        total_vision_tokens = sum(end - start for start, end in vision_ranges)
        total_text_tokens = len(tokens) - total_vision_tokens
        print(f"\nMask mode: {mask_mode}")
        if mask_mode == 'text':
            print(f"  → Will mask {total_text_tokens} text tokens (skipping {total_vision_tokens} vision tokens)")
        elif mask_mode == 'vision':
            print(f"  → Will mask {total_vision_tokens} vision tokens (skipping {total_text_tokens} text tokens)")
        else:
            print(f"  → Will mask ALL {len(tokens)} tokens ({total_text_tokens} text + {total_vision_tokens} vision)")
    
    # Initialize masker
    masker = VLAttentionMasker(model, processor, device)
    
    num_layers = len(model.model.language_model.layers)
    num_tokens = len(tokens)
    
    print(f"Number of layers: {num_layers}")
    print(f"Number of tokens: {num_tokens}")
    
    results = []
    
    # For each generation step (analyzing progressively longer sequences)
    current_input_ids = input_ids.clone()
    
    # Get EOS token ID(s) for early stopping
    eos_token_id = processor.tokenizer.eos_token_id
    im_end_token = "<|im_end|>"
    im_end_token_id = processor.tokenizer.encode(im_end_token, add_special_tokens=False)[-1] if im_end_token in processor.tokenizer.get_vocab() else None
    
    for gen_step in range(num_output_tokens):
        if gen_step > 0:
            # Generate next token
            with torch.no_grad():
                outputs = model(current_input_ids)
                logits = outputs.logits[0, -1, :]
                predicted_token_id = torch.argmax(logits).item()
                predicted_token = processor.tokenizer.decode([predicted_token_id])
                
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
        current_tokens = [processor.tokenizer.decode([token_id]) for token_id in current_input_ids[0]]
        current_num_tokens = len(current_tokens)
        
        if gen_step == 0:
            print(f"\nAnalyzing initial prompt ({current_num_tokens} tokens)...")
        else:
            print(f"Analyzing with {current_num_tokens} tokens (original + {gen_step} generated)...")
        
        # PRE-COMPUTE: Run forward pass once to get hidden states at each layer input
        print(f"Pre-computing hidden states for all {num_layers} layers...")
        cached_hidden_states = {}  # layer_idx -> hidden_states at input to that layer
        
        with torch.no_grad():
            hidden_states = model.model.language_model.embed_tokens(current_input_ids)
            batch_size, seq_length = current_input_ids.shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
            
            # Create proper attention mask (bidirectional for vision, causal for text)
            attention_mask = create_vl_attention_mask(seq_length, vision_ranges, device)
            
            # Get RoPE embeddings (MRoPE for VL, but text-only uses standard section)
            position_embeddings = model.model.language_model.rotary_emb(hidden_states, position_ids)
            
            # Cache initial embeddings as input to layer 0
            cached_hidden_states[0] = hidden_states.detach().clone()
            
            # Run through each layer and cache the output as input to next layer
            for idx in range(num_layers):
                if idx % 5 == 0 or idx == num_layers - 1:  # Print every 5 layers
                    print(f"  Pre-computing layer {idx}/{num_layers-1}...")
                
                layer = model.model.language_model.layers[idx]
                
                # Forward through layer
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                attn_output = layer.self_attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )[0]
                hidden_states = residual + attn_output
                
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states
                
                # Cache output as input to next layer
                if idx < num_layers - 1:
                    cached_hidden_states[idx + 1] = hidden_states.detach().clone()
        
        print(f"Pre-computation complete!\n")
        
        # Now analyze each layer
        for layer_idx in range(num_layers):
            print(f"Processing layer {layer_idx}/{num_layers-1}...")
            
            # Get final token position for this sequence
            final_pos = current_num_tokens - 1
            
            # Run baseline (using cached states)
            baseline_activations = masker.run_baseline_from_cache(
                cached_hidden_states[layer_idx],
                layer_idx,
                attention_mask,
                position_ids,
                position_embeddings
            )
            
            # For each token position, mask it and measure impact
            # Respect mask_mode: 'text', 'vision', or 'both'
            for mask_pos in range(current_num_tokens):
                is_vision_token = any(start <= mask_pos < end for start, end in vision_ranges)
                
                # Filter based on mask_mode
                if mask_mode == 'text' and is_vision_token:
                    continue  # Skip vision tokens in text-only mode
                elif mask_mode == 'vision' and not is_vision_token:
                    continue  # Skip text tokens in vision-only mode
                # If mask_mode == 'both', don't skip anything
                
                token_str = current_tokens[mask_pos]
                
                # Run with masking
                masked_activations = masker.run_masked_from_cache(
                    cached_hidden_states[layer_idx],
                    layer_idx,
                    attention_mask,
                    position_ids,
                    position_embeddings,
                    mask_position=mask_pos
                )
                
                # Extract updates at final position
                baseline_full = baseline_activations['resid_post'][0, final_pos, :] - baseline_activations['resid_pre'][0, final_pos, :]
                masked_full = masked_activations['resid_post'][0, final_pos, :] - masked_activations['resid_pre'][0, final_pos, :]
                
                baseline_attn = baseline_activations['attn_output'][0, final_pos, :]
                masked_attn = masked_activations['attn_output'][0, final_pos, :]
                
                # Variant 1: Full layer update
                l2_full, cos_full = compute_distances(baseline_full, masked_full)
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
                l2_attn, cos_attn = compute_distances(baseline_attn, masked_attn)
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
            outputs = model(generated_ids)
            logits = outputs.logits[0, -1, :]  # Get logits for final position
            predicted_token_id = torch.argmax(logits).item()
            predicted_token = processor.tokenizer.decode([predicted_token_id])
            generated_tokens.append(predicted_token)
            
            # Append predicted token for next iteration
            generated_ids = torch.cat([generated_ids, torch.tensor([[predicted_token_id]], device=device)], dim=1)
            
            # Show top-5 for first token
            if step == 0:
                top_k_values, top_k_indices = torch.topk(logits, k=5)
                top_k_tokens = [processor.tokenizer.decode([idx.item()]) for idx in top_k_indices]
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
        description='Analyze token masking impact on Qwen3-VL models (Phase 1/2a/2b: Text, Vision, or Both)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Text-only
  python mask_impact_vl.py --prompt "What is the capital of France?" --num-tokens 5
  
  # Phase 2a: Image + text (masking text only) - default
  python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --num-tokens 5
  
  # Phase 2b: Image + text (masking vision only) - NEW!
  python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --mask-mode vision --num-tokens 3
  
  # Phase 2c: Image + text (masking both)
  python mask_impact_vl.py --image testimg.png --prompt "What's in this image?" --mask-mode both --num-tokens 1
  
  # From prompt file
  python mask_impact_vl.py --prompt prompt.txt --num-tokens 10
  
  # Use different model
  python mask_impact_vl.py --prompt "test" --model "Qwen/Qwen3-VL-4B-Instruct" --device cuda
        """
    )
    
    parser.add_argument('--prompt', type=str, help='Text prompt or path to .txt file')
    parser.add_argument('--image', type=str, help='Path to image file (Phase 2: enables multimodal analysis)')
    parser.add_argument('--mask-mode', type=str, choices=['text', 'vision', 'both'], default='text',
                       help='Which tokens to mask: text (2a), vision (2b), or both (2c). Default: text')
    parser.add_argument('--num-tokens', type=int, default=1, help='Number of tokens to generate')
    parser.add_argument('--output', type=str, default='vl_masking_results', help='Output file basename')
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-VL-4B-Instruct", help='Model path')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
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
            prompt = "What is the capital of France?"
    
    # Run experiment
    df = run_vl_masking_experiment(
        prompt,
        model_path=args.model,
        device=device,
        num_output_tokens=args.num_tokens,
        image_path=args.image,
        mask_mode=args.mask_mode
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

