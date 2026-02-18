import torch
import os
import re

def save_checkpoint(state, is_best, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save epoch checkpoint
    epoch = state["epoch"]
    epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth")
    torch.save(state, epoch_path)

def get_latest_checkpoint(checkpoint_dir):
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
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    best_val_loss = checkpoint["best_val_loss"]

    return start_epoch, train_losses, val_losses, best_val_loss
