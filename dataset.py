import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    return train_transform, test_transform


def get_datasets(data_dir="./data"):
    train_transform, test_transform = get_transforms()

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    return full_train_dataset, test_dataset


def split_train_validation(full_train_dataset):
    num_total = len(full_train_dataset)
    num_val = 5000
    num_train = num_total - num_val

    train_indices = list(range(0, num_train))
    val_indices = list(range(num_train, num_total))

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(
        datasets.CIFAR10(
            root=full_train_dataset.root,
            train=True,
            download=False,
            transform=get_transforms()[1]  # no augmentation
        ),
        val_indices
    )

    return train_dataset, val_dataset


def get_dataloaders(batch_size=128, num_workers=2):
    full_train_dataset, test_dataset = get_datasets()
    train_dataset, val_dataset = split_train_validation(full_train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print(len(train_loader.dataset))  # 45000
    print(len(val_loader.dataset))    # 5000
    print(len(test_loader.dataset))   # 10000
