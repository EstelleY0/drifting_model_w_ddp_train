import argparse
from argparse import Namespace
from pathlib import Path

import torch
import yaml

from data.image_loader import get_cifar10_loader, get_mnist_loader
from model.encoder import create_feature_encoder, pretrain_mae

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Pre-train MAE Encoder")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_classes_per_batch", type=int, default=10)
    parser.add_argument("--n_samples_per_class", type=int, default=64)
    parser.add_argument("--mae_epochs", type=int, default=50)
    parser.add_argument("--mae_lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--save_dir", type=str, default="./pretrained")

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    args = Namespace(**config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        loader = get_cifar10_loader(n_classes_per_batch=args.n_classes_per_batch,
                                    n_samples_per_class=args.n_samples_per_class)
    elif args.dataset == "mnist":
        loader = get_mnist_loader(n_classes_per_batch=args.n_classes_per_batch,
                                  n_samples_per_class=args.n_samples_per_class)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    phi = create_feature_encoder(dataset=args.dataset).to(device)

    print(f"==> Starting MAE Pre-training on {args.dataset} for {args.mae_epochs} epochs...")

    trained_phi = pretrain_mae(
        feature_encoder=phi,
        train_loader=loader,
        num_epochs=args.mae_epochs,
        lr=args.mae_lr,
        mask_ratio=args.mask_ratio,
        device=device
    )

    model_path = save_path / f"mae_{args.dataset}_phi.pt"
    torch.save(trained_phi.state_dict(), model_path)
    print(f"==> Pre-trained encoder saved to {model_path}")

if __name__ == "__main__":
    main()
