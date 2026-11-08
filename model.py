"""Llama-style decoder: pre-norm RMSNorm, RoPE, SwiGLU, no biases, tied input/output embeddings.

The depth variants differ only in `n_layer`. For the energy-head work:
    h = model.hidden(idx)      # (B, T, d_model) final hidden states, after the last RMSNorm
    logits = model.lm_head(h)  # lm_head.weight is the output matrix W_U (tied to the embedding)
`forward` is exactly lm_head(hidden(idx)).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layer: int = 12
    d_model: int = 768
    n_head: int = 12
    d_ff: int = 2048  # SwiGLU hidden size, 8/3 * d_model
    vocab_size: int = 50304  # Phi-1.5 tokenizer (50,295 tokens) padded to a multiple of 64
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


def rope_tables(cfg):
    head_dim = cfg.d_model // cfg.n_head
    inv_freq = 1.0 / cfg.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(cfg.max_seq_len).float(), inv_freq)  # (T, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate (B, n_head, T, head_dim) queries/keys by position, computed in fp32."""
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_tables(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # One init for every depth (no depth-dependent scaling), so variants differ only in n_layer.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied embedding counted once

    def hidden(self, idx):
        """Final hidden states (B, T, d_model) after the last RMSNorm: the input to lm_head."""
        T = idx.size(1)
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max_seq_len {self.cfg.max_seq_len}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, idx, targets=None):
        """Logits (B, T, vocab_size), or mean next-token cross-entropy if targets are given (-1 = ignore)."""
        logits = self.lm_head(self.hidden(idx))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

    @classmethod
    def from_checkpoint(cls, path, device):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(ModelConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, ckpt
