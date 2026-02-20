import torch
import matplotlib.pyplot as plt

def plot_losses(x, y, title, x_label, y_label, save_path="./"):
    """
    Plot and save training and validation curves across epochs.

    Args:
        x (list or sequence): Metric values for training (e.g., loss or accuracy).
        y (list or sequence): Metric values for validation.
        title (str): Title of the plot.
        x_label (str): Label for the training curve in the legend.
        y_label (str): Label for the validation curve in the legend.
        save_path (str): Path to save the generated figure.
    """
    epochs = range(1, len(x) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, x, label=x_label)
    plt.plot(epochs, y, label=y_label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    

if __name__ == "__main__":
    checkpoint = torch.load("checkpoints/last.pth", map_location="cpu")
    train_losses = checkpoint["train_losses"]
    train_accs = checkpoint["train_accs"]
    val_losses = checkpoint["val_losses"]
    val_accs = checkpoint["val_accs"]
    
    plot_losses(
        train_losses,
        val_losses,
        "Training and Validation Loss",
        "Train Loss",
        "Validation Loss",
        save_path="loss_curve.png"
    )
    plot_losses(
        train_accs,
        val_accs,
        "Training and Validation Accuracy",
        "Train Accuracy",
        "Validation Accuracy",
        save_path="accuracy_curve.png"
    )
