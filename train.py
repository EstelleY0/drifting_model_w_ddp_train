import argparse
import os
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.image_loader import get_cifar10_loader, get_mnist_loader
from data.toy_data import get_toy_loader
from model.driftingModel import DriftingModel
from model.encoder import create_feature_encoder
from model.loss import ClassConditionalDriftingLoss
from utils.eval_utils import save_grid
from utils.train_utils import EMA, CheckpointManager


def setup(rank, world_size, gpu_id):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(gpu_id)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, config, gpu_ids):
    gpu_id = gpu_ids[rank]
    setup(rank, world_size, gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Determine if the dataset is 2D toy data
    is_toy = config.dataset in ["swiss_roll", "moons", "circles", "checkerboard"]

    # Model Initialization
    # For toy data, img_size is irrelevant but kept for structure consistency
    model = DriftingModel(
        img_size=32 if not is_toy else 1,
        in_channels=3 if not is_toy else 2,
        patch_size=4 if not is_toy else 1,
        dim=config.dim,
        depth=config.depth,
        num_heads=config.num_heads
    ).to(device)

    # Feature extractor setup
    # Uses pixel space (flatten) for toy data by passing None or specific flag
    phi = None if is_toy else create_feature_encoder(dataset=config.dataset).to(device)
    if phi is not None:
        phi.eval()

    model = DDP(model, device_ids=[gpu_id])

    # Loss, Optimizer, and EMA
    criterion = ClassConditionalDriftingLoss(phi=phi, use_pixel_space=is_toy).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    ema = EMA(model.module, decay=config.ema_decay)
    warmup_steps = config.warmup_steps
    base_lr = config.lr

    # Data Loader setup
    if is_toy:
        loader = get_toy_loader(name=config.dataset, batch_size=config.batch_size)
    elif config.dataset == "cifar10":
        loader = get_cifar10_loader(n_classes_per_batch=config.n_classes_per_batch,
                                    n_samples_per_class=config.n_samples_per_class)
    elif config.dataset == "mnist":
        loader = get_mnist_loader(n_classes_per_batch=config.n_classes_per_batch,
                                  n_samples_per_class=config.n_samples_per_class)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    writer = SummaryWriter(Path(config.log_dir) / "logs") if rank == 0 else None
    global_step = 0

    for epoch in range(config.epochs):
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)

        model.train()
        pbar = tqdm(loader) if rank == 0 else loader

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Adapt 2D coordinates (B, 2) for the PatchEmbed layer (B, C, H, W)
            if is_toy:
                images = images.view(images.size(0), 1, 1, 2)

            if global_step < warmup_steps:
                curr_lr = base_lr * (global_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_lr

            eps = torch.randn(images.size(0), config.latent_dim, device=device)
            alpha = torch.ones(images.size(0), device=device)
            x_gen = model(eps, labels, alpha)

            # Loss calculation: handle flattened features for toy data
            target_images = images.flatten(1) if is_toy else images
            pred_images = x_gen.flatten(1) if is_toy else x_gen
            loss, info = criterion(pred_images, labels, target_images, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update(model.module)
            global_step += 1

            if rank == 0:
                pbar.set_description(f"Epoch {epoch} | Loss: {info['loss']:.4f} | V_norm: {info['drift_norm']:.4f}")
            if global_step % 10 == 0:
                    writer.add_scalar('Loss/train', loss.item(), global_step)

        # Checkpointing and Visualization
        if rank == 0 and (epoch + 1) % config.save_freq == 0:
            ckpt_path = Path(config.log_dir) / "models" / f"epoch_{epoch + 1}.pt"
            CheckpointManager.save(ckpt_path, model.module, ema, optimizer, epoch + 1, global_step)

            with torch.no_grad():
                ema.shadow.eval()
                res_path = Path(config.log_dir) / "results" / f"epoch_{epoch + 1}.png"

                if is_toy:
                    # Visualization for 2D Toy data: Scatter Plot
                    test_eps = torch.randn(5000, config.latent_dim, device=device)
                    test_labels = torch.zeros(5000, dtype=torch.long, device=device)
                    samples = ema.shadow(test_eps, test_labels, torch.ones(5000, device=device))
                    samples = samples.cpu().numpy()

                    real_batch, _ = next(iter(loader))
                    real_samples = real_batch.cpu().numpy()

                    plt.figure(figsize=(6, 6))

                    plt.scatter(real_samples[:, 0], real_samples[:, 1],
                                color='red',
                                alpha=0.3,
                                s=3,
                                label='Real Data (GT)')

                    plt.scatter(samples[:, 0], samples[:, 1],
                                color='blue',
                                alpha=0.5,
                                s=2,
                                label='Generated')

                    plt.xlim(-3, 3)
                    plt.ylim(-3, 3)
                    plt.title(f"Epoch {epoch + 1}")
                    plt.savefig(res_path)
                    plt.close()
                else:
                    # Visualization for Image data: Image Grid
                    test_eps = torch.randn(16, *images.shape[1:], device=device)
                    test_labels = torch.arange(10, device=device).repeat(2)[:16]
                    samples = ema.shadow(test_eps, test_labels, torch.ones(16, device=device))
                    save_grid(samples, res_path, nrow=4)

    if rank == 0: writer.close()
    cleanup()


def load_config_and_args():
    """Merge YAML config with command line arguments"""
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gpu_idx", type=str)
    parser.add_argument("--epochs", type=int)

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    conf = Namespace(**config)
    return conf

def main():
    config = load_config_and_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    gpu_ids = [int(i) for i in config.gpu_idx.split(",")]
    world_size = len(gpu_ids)

    # Directory preparation
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "models").mkdir(parents=True, exist_ok=True)
    (log_path / "results").mkdir(parents=True, exist_ok=True)
    (log_path / "logs").mkdir(parents=True, exist_ok=True)

    print(f"Starting Training: {config.exp_name} on {world_size} GPUs")
    mp.spawn(main_worker, args=(world_size, config, gpu_ids), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
