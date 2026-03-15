import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ToyDataset2D(Dataset):
    """
    Toy 2D datasets for testing and visualization
    Includes swiss_roll, checkerboard, circles, moons, and gaussian_mixture
    """
    def __init__(self, name="swiss_roll", n_samples=10000, noise=0.0, seed=42):
        super().__init__()
        self.name = name
        self.n_samples = n_samples
        self.noise = noise
        np.random.seed(seed)

        if name == "swiss_roll":
            self.data = self._make_swiss_roll()
        elif name == "checkerboard":
            self.data = self._make_checkerboard()
        elif name == "circles":
            self.data = self._make_circles()
        elif name == "moons":
            self.data = self._make_moons()
        elif name == "gaussian_mixture":
            self.data = self._make_gaussian_mixture()
        else:
            raise ValueError(f"Unknown dataset: {name}")

        # Standardize features to zero mean and unit variance
        self.data = (self.data - self.data.mean(axis=0)) / self.data.std(axis=0)
        self.data = torch.from_numpy(self.data).float()

    def _make_swiss_roll(self):
        t = 1.5 * np.pi * (1 + 2 * np.random.rand(self.n_samples))
        x, y = t * np.cos(t), t * np.sin(t)
        data = np.stack([x, y], axis=1)
        data += self.noise * np.random.randn(*data.shape)
        return data

    def _make_checkerboard(self):
        x1 = np.random.rand(self.n_samples) * 4 - 2
        x2 = np.random.rand(self.n_samples) - np.random.randint(0, 2, self.n_samples) * 2
        x2 += (np.floor(x1) % 2).astype(np.float32)
        data = np.stack([x1, x2], axis=1) * 2
        data += self.noise * np.random.randn(*data.shape)
        return data

    def _make_circles(self):
        n_inner = self.n_samples // 2
        n_outer = self.n_samples - n_inner
        t_inner = 2 * np.pi * np.random.rand(n_inner)
        r_inner = 0.5 + self.noise * np.random.randn(n_inner)
        x_i, y_i = r_inner * np.cos(t_inner), r_inner * np.sin(t_inner)
        t_outer = 2 * np.pi * np.random.rand(n_outer)
        r_outer = 1.5 + self.noise * np.random.randn(n_outer)
        x_o, y_o = r_outer * np.cos(t_outer), r_outer * np.sin(t_outer)
        return np.stack([np.concatenate([x_i, x_o]), np.concatenate([y_i, y_o])], axis=1)

    def _make_moons(self):
        n_half = self.n_samples // 2
        t_u = np.pi * np.random.rand(n_half)
        x_u, y_u = np.cos(t_u), np.sin(t_u)
        t_l = np.pi * np.random.rand(n_half)
        x_l, y_l = 1 - np.cos(t_l), -np.sin(t_l) - 0.5
        data = np.stack([np.concatenate([x_u, x_l]), np.concatenate([y_u, y_l])], axis=1)
        data += self.noise * np.random.randn(*data.shape)
        return data

    def _make_gaussian_mixture(self, n_components=8):
        samples_per_comp = self.n_samples // n_components
        data = []
        for i in range(n_components):
            angle = 2 * np.pi * i / n_components
            center = np.array([2 * np.cos(angle), 2 * np.sin(angle)])
            data.append(center + 0.2 * np.random.randn(samples_per_comp, 2))
        return np.concatenate(data, axis=0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], 0

def get_toy_loader(name="swiss_roll", n_samples=10000, batch_size=128):
    dataset = ToyDataset2D(name=name, n_samples=n_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
