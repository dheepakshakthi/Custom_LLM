import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from collections import OrderedDict

# --- Configuration Loading ---
def load_config(config_path='config.json'):
    """Loads model configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class ModelArgs:
    """A simple class to hold model arguments, loaded from the config."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- Core Building Blocks ---

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Normalizes the hidden states for stable training.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    """
    SwiGLU activation function. It provides a gating mechanism
    which helps in better information flow and has been shown to
    improve performance over standard ReLU or GeLU.
    """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).
    Encodes absolute positional information with a rotation matrix.
    This is more effective than learned or sinusoidal position embeddings
    for capturing relative positional information in sequences.
    """
    def __init__(self, dim, max_position_embeddings=4096, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    GQA strikes a balance between Multi-Head Attention (MHA) and
    Multi-Query Attention (MQA). It provides much of the speed of MQA
    while retaining the quality of MHA, making it ideal for our resource
    constrained goal.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_position_embeddings=args.max_seq_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, position_ids: torch.Tensor):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        cos, sin = self.rotary_emb(xv, seq_len=seqlen)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, position_ids)

        # GQA logic: repeat K and V heads to match Q heads
        xk = xk.repeat_interleave(self.n_rep, dim=2)
        xv = xv.repeat_interleave(self.n_rep, dim=2)

        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention for efficiency (Flash Attention)
        # This is a key optimization for memory and speed
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    """
    A single block of the Transformer.
    [RMSNorm] -> [Attention] -> [Residual] -> [RMSNorm] -> [FFN] -> [Residual]
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)
        self.feed_forward = nn.Sequential(
            nn.Linear(args.dim, args.hidden_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(args.hidden_dim, args.dim, bias=False),
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, position_ids: torch.Tensor):
        # Attention block with pre-normalization
        h = x + self.attention(self.attention_norm(x), mask, position_ids)
        # Feed-forward block with pre-normalization
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class CogniMamba(nn.Module):
    """
    The main Cogni-Mamba Language Model class.
    """
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.layers = nn.ModuleList()
        for _ in range(params.n_layers):
            self.layers.append(TransformerBlock(params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # Tie weights for efficiency
        self.tok_embeddings.weight = self.output.weight

        # Causal attention mask
        self.mask = self.build_causal_mask(params.max_seq_len)

    def build_causal_mask(self, max_seq_len):
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, tokens: torch.Tensor):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        position_ids = torch.arange(0, seqlen, dtype=torch.long, device=tokens.device)

        # Ensure mask is on the correct device
        mask = self.mask.to(h.device)[:seqlen, :seqlen]

        for layer in self.layers:
            h = layer(h, mask, position_ids)

        h = self.norm(h)
        output = self.output(h)
        return output

    def get_num_params(self):
        """Calculates and returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Load configuration and create model arguments object
    config = load_config()
    model_args = ModelArgs(**config)

    # 2. Instantiate the model
    model = CogniMamba(model_args)

    # 3. Print model architecture and parameter count
    print("168 Model Architecture:\n", model)
    num_params = model.get_num_params()
    print(f"\nTotal Trainable Parameters: {num_params / 1e9:.3f} Billion")

    # 4. Example forward pass
    print("\n--- Running a test forward pass ---")
    # Create a dummy input tensor
    # (batch_size, sequence_length)
    dummy_input = torch.randint(0, model_args.vocab_size, (1, 128))

    try:
        # Move model to GPU if available
        if torch.cuda.is_available():
            print("Moving model to GPU...")
            model.to('cuda')
            dummy_input = dummy_input.to('cuda')

        # Perform the forward pass
        output = model(dummy_input)
        print("Forward pass successful!")
        print("Output shape:", output.shape) # (batch_size, seq_len, vocab_size)
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")
