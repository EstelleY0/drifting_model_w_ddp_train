import torch.nn as nn
import torch.nn.functional as F

from model.layers import RMSNorm, apply_rope, modulate, SwiGLU


class Attention(nn.Module):
    """
    Multi-head attention mechanism with QK-Norm and Flash Attention optimization
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, use_qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Apply RMSNorm to Query and Key for training stability
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, N, C = x.shape

        # Project and separate into q, k, v tensors
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Optional normalization of Query and Key tensors
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # Apply Rotary Position Embedding: q = (q * cos) + (rotate_half(q) * sin)
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        # Rearrange to (batch, heads, sequence, dim) for optimized attention kernels
        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        # Memory-efficient scaled dot-product attention
        x = F.scaled_dot_product_attention(q, k, v)

        # Restore original shape and apply final linear projection
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DriftingBlock(nn.Module):
    """
    Transformer block utilizing adaLN-Zero for conditioning
    Modulates input features using shift, scale, and gating parameters
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, use_qk_norm=True):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)

        self.norm2 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden, dim)

        # Regression of 6 modulation parameters from the conditioning vector c
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def forward(self, x, c, rope_cos=None, rope_sin=None):
        # Extract 6 modulation parameters: shift, scale, and gate for both Attention and MLP
        modulation = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        # Self-attention path with adaptive layer normalization and gating
        # x = x + gate * Attention(LayerNorm(x) * (1 + scale) + shift)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope_cos,
            rope_sin,
        )

        # MLP path with adaptive layer normalization and gating
        # x = x + gate * MLP(LayerNorm(x) * (1 + scale) + shift)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """
    Final output layer with adaLN modulation and linear projection back to patch space
    """

    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)

        # Regresses 2 parameters for final modulation: shift and scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        # Apply final modulation and project to image channels
        # x = (LayerNorm(x) * (1 + scale) + shift) @ W + b
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x
