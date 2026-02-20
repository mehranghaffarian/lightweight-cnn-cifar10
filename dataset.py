import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_train_transforms():
    """
    Define data augmentations and normalization used during training.

    Returns:
        torchvision.transforms.Compose: Training data transformations.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    return train_transform

def get_test_transform():    
    """
    Define normalization applied to validation and test data.

    Returns:
        torchvision.transforms.Compose: Evaluation data transformations.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])


def get_train_dataset(data_dir="./data"):
    """
    Load the CIFAR-10 training dataset with training-time augmentations.

    Args:
        data_dir (str): Directory to store or load the dataset.

    Returns:
        torchvision.datasets.CIFAR10: Full training dataset.
    """
    train_transform = get_train_transforms()

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    return full_train_dataset

def get_test_dataset(data_dir="./data"):
    """
    Load the CIFAR-10 test dataset with evaluation preprocessing.

    Args:
        data_dir (str): Directory to store or load the dataset.

    Returns:
        torchvision.datasets.CIFAR10: Test dataset.
    """
    test_transform = get_test_transform()
    
    return datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )


def split_train_validation(full_train_dataset):
    """
    Split the training dataset into training and validation subsets.

    Args:
        full_train_dataset (Dataset): Full CIFAR-10 training dataset.

    Returns:
        tuple: Training and validation datasets.
    """
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
            transform=get_test_transform()  # no augmentation
        ),
        val_indices
    )

    return train_dataset, val_dataset


def get_train_dataloaders(batch_size=128, num_workers=2):
    """
    Create DataLoaders for training and validation.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: Training and validation DataLoaders.
    """
    full_train_dataset = get_train_dataset()
    train_dataset, val_dataset = split_train_validation(full_train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader

def get_test_dataloader(batch_size=128, num_workers=2):
    """
    Create a DataLoader for the test dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: Test DataLoader.
    """
    test_dataset = get_test_dataset()
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    train_loader, val_loader = get_train_dataloaders()
    test_loader = get_test_dataloader()

    print(len(train_loader.dataset))  # 45000
    print(len(val_loader.dataset))    # 5000
    print(len(test_loader.dataset))   # 10000
