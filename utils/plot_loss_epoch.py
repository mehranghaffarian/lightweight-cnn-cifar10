import torch
import matplotlib.pyplot as plt

def plot_losses(x, y, title, x_label, y_label, save_path="./"):
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
