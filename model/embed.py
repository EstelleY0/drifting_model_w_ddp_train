import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    Divides input images into small patches and projects them into a latent space
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 2D convolution creates embeddings of shape (batch, embed_dim, height/patch, width/patch)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Flatten spatial dimensions and transpose for Transformer compatibility
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LabelEmbedder(nn.Module):
    """
    Class Label Embedding with Classifier-Free Guidance support
    Utilizes an extra token for the null/unconditional class
    """

    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        # Table size includes +1 for the null condition used in guidance
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Randomly replaces labels with the null class index to train for guidance
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool()

        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=True, force_drop_ids=None):
        if self.dropout_prob > 0 and train:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class AlphaEmbedder(nn.Module):
    """
    CFG Scale alpha Embedding using Fourier features
    Maps the scalar strength parameter to a high-dimensional vector
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def fourier_features(alpha, dim, max_period=10.0):
        """
        Generates sinusoidal embeddings for the guidance scale
        encoding = [cos(alpha * freq), sin(alpha * freq)]
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=alpha.device) / half)
        args = alpha[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, alpha):
        # Transform scalar scale alpha into a dense conditioning vector
        fourier = self.fourier_features(alpha, self.frequency_embedding_size)
        return self.mlp(fourier)


class StyleEmbedder(nn.Module):
    """
    Random Style Embeddings for increased distribution coverage
    Draws tokens from a learnable codebook to form a joint distribution with noise
    """

    def __init__(self, hidden_size, num_tokens=32, codebook_size=64):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, hidden_size)

    def forward(self, batch_size, device):
        # Generate random indices to pick style tokens from the codebook
        indices = torch.randint(0, self.codebook_size, (batch_size, self.num_tokens), device=device)
        embeddings = self.codebook(indices)

        # Aggregate style tokens into a single conditioning vector
        # style = sum(tokens)
        style = embeddings.sum(dim=1)
        return style
