import torch
import torch.nn as nn

from model.blocks import DriftingBlock, FinalLayer
from model.embed import PatchEmbed, LabelEmbedder, StyleEmbedder, AlphaEmbedder
from model.layers import RoPE


class DriftingModel(nn.Module):
    """
    Implementation of the Drifting Model architecture
    Maps a prior distribution to a data distribution via a single-pass pushforward: q = f#p_prior
    """

    def __init__(
            self,
            img_size=32,
            patch_size=4,
            in_channels=3,
            dim=256,
            depth=6,
            num_heads=4,
            mlp_ratio=4.0,
            num_classes=10,
            label_dropout=0.1,
            in_context_tokens=16,
            use_style_embed=True,
            latent_dim=16
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim  # Feature dimension D
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.latent_dim=latent_dim

        # Project image patches into latent embeddings
        if self.img_size > 1:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=dim,
            )
        else:
            self.latent_proj = nn.Linear(latent_dim, dim)

        # Prependable learnable tokens for in-context information
        self.in_context_tokens = in_context_tokens
        self.in_context_tokens = nn.Parameter(
            torch.randn(1, in_context_tokens, dim) * 0.02
        )

        # Positional encoding using Rotary Position Embedding
        head_dim = dim // num_heads
        self.rope = RoPE(
            dim=head_dim,
            max_seq_len=self.num_patches + in_context_tokens + 64,
        )

        # Conditioning vectors for class, CFG scale alpha, and style
        self.label_embed = LabelEmbedder(num_classes, dim, label_dropout)
        self.alpha_embed = AlphaEmbedder(dim)
        self.use_style_embed = use_style_embed
        if use_style_embed:
            self.style_embed = StyleEmbedder(dim)

        # Sequence of transformer blocks with adaptive layer normalization
        self.blocks = nn.ModuleList([
            DriftingBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qk_norm=True,
            )
            for _ in range(depth)
        ])

        # Map latent tokens back to the target patch space
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self._init_weights()

    def _init_weights(self):
        """
        Weight initialization focusing on adaLN-Zero stability
        Modulation layers are initialized to zero to act as an identity at the start
        """

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)

        # Initialize gate and scale modulations to zero
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)

        # Linear output layer initialization
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

        if hasattr(self, 'latent_proj'):
            nn.init.xavier_uniform_(self.latent_proj.weight)
            nn.init.zeros_(self.latent_proj.bias)

    def unpatchify(self, x):
        """
        Reshape tokens back to image spatial dimensions
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p

        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(self, eps, labels, alpha, force_drop_ids=None):
        """
        Forward pushforward operation: x = f(eps, labels, alpha)
        eps: Input noise from prior distribution
        labels: Conditional class labels
        alpha: Classifier-Free Guidance scale
        """
        B = eps.shape[0]
        device = eps.device

        # Latent projection of input noise epsilon
        if self.img_size > 1:
            x = self.patch_embed(eps)
        else:
            x = self.latent_proj(eps)
            x = x.unsqueeze(1)

        # Prepend in-context tokens to the sequence: [tokens, patches]
        register = self.in_context_tokens.expand(B, -1, -1)
        x = torch.cat([register, x], dim=1)

        # Apply positional rotation
        if self.img_size > 1:
            seq_len = x.shape[1]
            rope_cos, rope_sin = self.rope(x, seq_len)
        else:
            rope_cos, rope_sin = None, None

        # Construct conditioning vector c = label_emb + alpha_emb + style_emb
        c = self.label_embed(labels, self.training, force_drop_ids)
        c = c + self.alpha_embed(alpha)
        if self.use_style_embed:
            c = c + self.style_embed(B, device)

        # Pass through the transformer backbone
        for block in self.blocks:
            x = block(x, c, rope_cos, rope_sin)

        # Exclude in-context tokens for final projection
        num_registers = self.in_context_tokens.shape[1]
        x = x[:, num_registers:, :]

        x = self.final_layer(x, c)

        if self.img_size > 1:
            x = self.unpatchify(x)
        else:
            x = x.squeeze(1)

        return x

    def forward_with_cfg(self, eps, labels, alpha=1.0):
        """
        Classifier-Free Guidance inference:
        output = uncond + alpha * (cond - uncond)
        """
        B = eps.shape[0]
        device = eps.device

        alpha_tensor = torch.full((B,), alpha, device=device, dtype=eps.dtype)

        # Parallel execution of conditional and unconditional paths
        e_combined = torch.cat([eps, eps], dim=0)
        labels_combined = torch.cat([labels, labels], dim=0)
        alpha_combined = torch.cat([alpha_tensor, alpha_tensor], dim=0)

        # Force label dropout for the unconditional branch
        force_drop = torch.cat([
            torch.zeros(B, device=device),
            torch.ones(B, device=device),
        ]).bool()

        out = self.forward(e_combined, labels_combined, alpha_combined, force_drop)

        # Extrapolate between distributions
        cond, uncond = out.chunk(2, dim=0)
        return uncond + alpha * (cond - uncond)
