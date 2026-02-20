import torch
import os
import re

def save_checkpoint(state, checkpoint_dir="checkpoints"):
    """
    Save a training checkpoint for the current epoch.

    Args:
        state (dict): Training state containing model/optimizer states and epoch.
        checkpoint_dir (str): Directory where checkpoints are stored.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save epoch checkpoint
    epoch = state["epoch"]
    epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
    torch.save(state, epoch_path)

def get_latest_checkpoint(checkpoint_dir):
    """
    Find the most recent checkpoint file based on epoch number.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files.

    Returns:
        str or None: Path to the latest checkpoint, or None if none exist.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    pattern = re.compile(r"epoch_(\d+)\.pth")
    max_epoch = -1
    latest_path = None

    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                latest_path = os.path.join(checkpoint_dir, filename)

    return latest_path

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load model and optimizer states from a checkpoint and resume training.

    Args:
        model (torch.nn.Module): Model to restore.
        optimizer (torch.optim.Optimizer): Optimizer to restore.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device for loading the checkpoint.

    Returns:
        tuple: Next epoch index and stored training/validation metrics.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    train_losses = checkpoint["train_losses"]
    train_accs = checkpoint["train_accs"]
    val_losses = checkpoint["val_losses"]
    val_accs = checkpoint["val_accs"]

    return start_epoch, train_losses, train_accs, val_losses, val_accs
