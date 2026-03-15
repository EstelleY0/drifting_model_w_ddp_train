import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data.sampler import ClassBalancedSampler


def get_mnist_loader(root="./data", train=True, n_classes_per_batch=10, n_samples_per_class=32):
    """
    MNIST loader with class balancing and [-1, 1] normalization
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)

    sampler = ClassBalancedSampler(
        dataset.targets,
        n_classes_per_batch=n_classes_per_batch,
        n_samples_per_class=n_samples_per_class
    )

    return DataLoader(
        dataset,
        batch_size=n_classes_per_batch * n_samples_per_class,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )


def get_cifar10_loader(root="./data", train=True, n_classes_per_batch=10, n_samples_per_class=32):
    """
    CIFAR-10 loader with class balancing and [-1, 1] normalization
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    sampler = ClassBalancedSampler(
        dataset.targets,
        n_classes_per_batch=n_classes_per_batch,
        n_samples_per_class=n_samples_per_class
    )

    return DataLoader(
        dataset,
        batch_size=n_classes_per_batch * n_samples_per_class,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
