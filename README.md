# Generative Modeling via Drifting (PyTorch Implementation)

This project is a PyTorch implementation of [**Generative Modeling via Drifting**](https://arxiv.org/abs/2602.04770). It features a generative model framework that maps a prior noise distribution to a data distribution via a single-pass pushforward operation ($x = f(\epsilon, c)$). 

The model learns by calculating a **Drifting Field** between real samples and generated samples, supporting both high-dimensional images (MNIST, CIFAR-10) and 2D toy datasets (Swiss roll, Moons, etc.).

---

## 1. Project Structure

```text
.
├── config.yaml          # Centralized hyperparameter and experiment management
├── train.py             # Main DDP-based training script
├── model/
│   ├── drifting.py      # Core Drifting Field V and normalization (S_j, lambda_j)
│   ├── driftingModel.py # DiT-based Transformer architecture
│   ├── loss.py          # Class-Conditional Drifting Loss implementation
│   ├── blocks.py        # Transformer blocks (Attention, DriftingBlock)
│   ├── layers.py        # RMSNorm, RoPE, SwiGLU layers
│   ├── embed.py         # Patch, Label, Style, and Alpha embeddings
│   └── encoder.py       # Feature extractor (Phi) and MAE pre-training
├── data/
│   ├── toy_data.py      # 2D coordinate datasets (Swiss roll, Moons, etc.)
│   ├── image_loader.py  # MNIST and CIFAR-10 data loaders
│   └── sampler.py       # Class-balanced sampler (Section A.8)
└── utils/
    ├── utils.py         # EMA and checkpoint management
    └── eval_utils.py    # Result visualization and grid generation
```

---

## 2. Training Guide

### Configuration
All hyperparameters, dataset selections, and hardware configurations are managed through the `config.yaml` file. 

### Running the Training
To initiate training using the default settings in your YAML file, execute:
```bash
python3 train.py

# Train CIFAR-10
python3 train --dataser cifar10

# Train MNIST
python3 train --dataset mnist
```

You can override any parameter directly via command line arguments:
```bash
python3 train.py --dataset swiss_roll --lr 1e-4 --gpu_idx "0,1" ...
```

---

## 3. Outputs and Logging

Training artifacts are organized in the `log_dir` defined in your configuration:

* **`models/`**: Stores checkpoints (`.pt`) at every `save_freq`. Each file contains:
    * Raw model weights and **EMA (Exponential Moving Average)** shadow weights.
    * Optimizer and scheduler states.
    * Training metadata (epoch and global step).
* **```results/```**: Contains visual samples generated periodically to monitor progress.
    * **Image Data**: Saved as standardized image grids (`.png`).
    * **Toy Data**: Saved as 2D scatter plots to visualize distribution alignment.
* **```logs/```**: Stores Tensorboard event files for tracking loss curves and drift norms.

---

## 4. Inference and Generation

### Running the Inference Script
To generate samples using a trained checkpoint, use the `infer.py` script. This script automatically handles both image and 2D toy data formats based on your `config.yaml`.

```bash
# Basic inference (generates 16 samples by default)
python3 infer.py --ckpt ./outputs/models/epoch_200.pt

# Generate with higher guidance strength (alpha) for better quality
python3 infer.py --ckpt ./outputs/models/epoch_200.pt --alpha 2.0 --num_samples 64
```

### Loading the Model Programmatically
If you want to perform inference within a Python script, use the `CheckpointManager` to restore the **EMA shadow weights** for optimal generation quality.

```python
import torch
from model.driftingModel import DriftingModel
from utils.utils import CheckpointManager

# 1. Initialize architecture and load EMA weights
model = DriftingModel(img_size=32, dim=256, ...)
CheckpointManager.load("outputs/models/epoch_200.pt", model)
model.eval()

# 2. Prepare prior noise (epsilon) and class labels
# For CIFAR-10 (3 channels, 32x32)
eps = torch.randn(16, 3, 32, 32)      
labels = torch.randint(0, 10, (16,))  
alpha = torch.tensor([1.0] * 16)      

# 3. Single-pass generation (Pushforward)
with torch.no_grad():
    # Use forward_with_cfg for Classifier-Free Guidance extrapolation
    samples = model.forward_with_cfg(eps, labels, alpha=1.5)
```

### Output
* **Images**: Saved as grids in the `./inference_results` directory.
* **Toy Data**: Saved as scatter plots in the `./inference_results` directory.
---

## Reference

```bibtex
@article{deng2026generative,
  title={Generative Modeling via Drifting},
  author={Deng, Mingyang and Li, He and Li, Tianhong and Du, Yilun and He, Kaiming},
  journal={arXiv preprint arXiv:2602.04770},
  year={2026}
}
```