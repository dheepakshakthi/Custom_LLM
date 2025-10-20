"""
200M Parameter LLM Architecture
Inspired by SmolLM-135M but scaled up for better reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, seq_len: int):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE and Flash Attention support"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_flash: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV projection and split
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim)
        sin = sin[None, None, :, :]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        if self.use_flash:
            # Use Flash Attention if available (more efficient)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Standard attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation (better than GELU)"""
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU requires 2 projections for the gate mechanism
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(W1*x) ⊙ W3*x, then project back
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization (more stable training)"""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture (better for deep networks)
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class LLM200M(nn.Module):
    """
    200M Parameter Language Model
    
    Architecture inspired by SmolLM-135M but scaled for better reasoning:
    - 768 embedding dimension
    - 24 layers (deeper for better reasoning)
    - 12 attention heads
    - 3072 FFN intermediate dimension (4x embed_dim)
    - RoPE positional embeddings
    - SwiGLU activation
    - Pre-normalization
    
    Total parameters: ~200M
    """
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 tokenizer vocab size
        embed_dim: int = 768,
        num_layers: int = 24,
        num_heads: int = 12,
        ff_dim: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying (share embeddings with output layer)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {self.num_params:,} parameters")
        
    def _init_weights(self, module):
        """Initialize weights using scaled initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) optional
            labels: (batch_size, seq_len) optional, for computing loss
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar loss if labels provided
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1
    ) -> torch.Tensor:
        """
        Generate text using nucleus sampling
        
        Args:
            input_ids: (batch_size, seq_len)
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: keep only top k tokens
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for repeating tokens
            
        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get predictions for the last token
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we've exceeded max length
            if input_ids.shape[1] >= self.max_seq_len:
                break
        
        return input_ids


if __name__ == "__main__":
    # Test model
    model = LLM200M(vocab_size=50257)
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    labels = torch.randint(0, 50257, (batch_size, seq_len))
    
    logits, loss = model(input_ids, labels=labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, 50257, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
