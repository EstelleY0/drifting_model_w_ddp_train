import copy

import torch


class EMA:
    """
    Exponential Moving Average for model stability
    shadow = shadow.lerp(model, 1 - decay)
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        # Linear interpolation for parameter smoothing
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.lerp_(m_p.data, 1.0 - self.decay)


class CheckpointManager:
    """Simplified manager for saving/loading training state"""
    @staticmethod
    def save(path, model, ema, optimizer, epoch, step):
        torch.save({
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }, path)

    @staticmethod
    def load(path, model, ema=None):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if ema:
            ema.shadow.load_state_dict(ckpt["ema"])
        return ckpt
