import torch
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, save_path=None):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    

# checkpoint = torch.load("checkpoints/last.pth", map_location="cpu")
# train_losses = checkpoint["train_losses"]
# val_losses = checkpoint["val_losses"]

# plot_losses(
#     train_losses,
#     val_losses,
#     save_path="loss_curve.png"
# )
