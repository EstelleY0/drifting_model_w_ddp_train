import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from model.driftingModel import DriftingModel
from utils.eval_utils import save_grid
from utils.train_utils import CheckpointManager


def run_inference():
    parser = argparse.ArgumentParser(description="Drifting Model Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint .pt file")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Classifier-Free Guidance scale")
    parser.add_argument("--out_dir", type=str, default="./inference_results", help="Output directory")
    args = parser.parse_args()

    # Load config from original project
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model based on config
    is_toy = config['dataset'] in ["swiss_roll", "moons", "circles", "checkerboard"]
    model = DriftingModel(
        img_size=32 if not is_toy else 1,
        dim=config['dim'],
        depth=config['depth'],
        num_heads=config['num_heads']
    ).to(device)

    # Load Checkpoint
    # We load into EMA shadow for better quality
    CheckpointManager.load(args.ckpt, model)
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt}")

    # Generate Samples
    output_path = Path(args.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Prepare noise and labels
        if is_toy:
            eps = torch.randn(1000, 1, 1, 2, device=device)
            labels = torch.zeros(1000, dtype=torch.long, device=device)
        else:
            # 3 channels, 32x32 resolution for CIFAR/MNIST
            c = 1 if config['dataset'] == "mnist" else 3
            eps = torch.randn(args.num_samples, c, 32, 32, device=device)
            # Sample labels uniformly for visualization
            labels = torch.arange(config['n_classes_per_batch'], device=device).repeat(10)[:args.num_samples]

        # Inference with CFG
        samples = model.forward_with_cfg(eps, labels, alpha=args.alpha)

        # Save Results
        if is_toy:
            samples = samples.flatten(1).cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=2)
            plt.title(f"Generated {config['dataset']} (alpha={args.alpha})")
            save_path = output_path / f"generated_{config['dataset']}.png"
            plt.savefig(save_path)
            plt.close()
        else:
            save_path = output_path / f"generated_samples_alpha_{args.alpha}.png"
            save_grid(samples, str(save_path), nrow=int(args.num_samples**0.5))

    print(f"Inference complete. Results saved to {save_path}")

if __name__ == "__main__":
    run_inference()
