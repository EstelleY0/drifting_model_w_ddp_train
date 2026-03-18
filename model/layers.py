import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    rms = sqrt(mean(x^2) + eps)
    output = (x / rms) * weight
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function
    output = Linear(silu(Linear(x)) * Linear(x))
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    """
    Rotary Position Embedding
    Precomputes sinusoidal inverse frequencies for positional encoding
    """
    def __init__(self, dim, max_seq_len=1024, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._cache = None

    def _get_cache(self, seq_len, device, dtype):
        """
        Retrieves or builds cached cos and sin values for the given sequence length
        """
        if self._cache is not None and self._cache[0].shape[2] >= seq_len:
            return self._cache[0][..., :seq_len, :], self._cache[1][..., :seq_len, :]

        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        self._cache = (cos, sin)
        return cos, sin

    def forward(self, x, seq_len):
        cos, sin = self._get_cache(seq_len, x.device, x.dtype)
        return cos, sin


def rotate_half(x):
    """
    Rotates half of the hidden dimensions for rotary embedding
    [x1, x2] -> [-x2, x1]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """
    Applies rotary position embeddings to query and key tensors
    q_embed = (q * cos) + (rotate_half(q) * sin)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def modulate(x, shift, scale):
    """
    Applies adaptive layer normalization (adaLN) modulation
    output = x * (1 + scale) + shift
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
