import numpy as np
import torch
from scipy import linalg


@torch.no_grad()
def get_fid_stats(images, inception, device):
    """Compute mean and covariance for FID"""
    inception.eval().to(device)
    # Resize and normalize as required by InceptionV3
    images = torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear")
    images = (images + 1) / 2

    feats = []
    for i in range(0, len(images), 64):
        batch = images[i:i + 64].to(device)
        f = inception(batch)[0].view(batch.shape[0], -1)
        feats.append(f.cpu())

    feats = torch.cat(feats, dim=0).numpy()
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Standard FID formula with numerical stability"""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def save_grid(images, path, nrow=8):
    """Quick image grid saver"""
    from torchvision.utils import save_image
    # Scale back to [0, 1] for saving
    save_image(images, path, nrow=nrow, normalize=True, value_range=(-1, 1))
