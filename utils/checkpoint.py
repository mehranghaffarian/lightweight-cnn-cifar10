import torch
import os

def save_checkpoint(state, is_best, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save last checkpoint
    last_path = os.path.join(checkpoint_dir, f"epoch_{state.epoch:03d}.pth")
    torch.save(state, last_path)

    # Save best model separately
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    best_val_loss = checkpoint["best_val_loss"]

    return start_epoch, train_losses, val_losses, best_val_loss
