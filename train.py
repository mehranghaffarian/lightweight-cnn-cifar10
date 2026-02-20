import torch
import torch.nn as nn
import torch.optim as optim
import logging

from models.cnn import SmallResNet
from dataset import *
from utils.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.plot_loss_epoch import plot_losses

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for a single epoch.

    Performs forward pass, loss computation, backpropagation,
    and parameter updates over the entire training dataset.

    Args:
        model (nn.Module): Neural network model.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (torch.device): Training device.

    Returns:
        tuple: Average training loss and accuracy for the epoch.
    """
    model.train()

    total_loss = 0.0
    total_acc = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)

    return total_loss / len(loader), total_acc / len(loader)

def validate(model, loader, criterion, device):
    """
    Evaluate the model on the validation set.

    Runs inference without gradient computation to measure
    generalization performance.

    Args:
        model (nn.Module): Neural network model.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Evaluation device.

    Returns:
        tuple: Average validation loss and accuracy.
    """
    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels)

    return total_loss / len(loader), total_acc / len(loader)

def train():
    """
    Full training pipeline for CIFAR-10 classification.

    - Initializes model, optimizer, scheduler, and loss
    - Loads latest checkpoint if available
    - Trains for a fixed number of epochs
    - Saves model checkpoints after each epoch
    - Tracks training and validation loss/accuracy

    Returns:
        tuple:
            - model (nn.Module): Trained model
            - train_losses (list): Training loss per epochs
            - train_accs (list): Training accuracy per epochs
            - val_losses (list): Validation loss per epochs
            - val_accs (list): Validation accuracy per epochs
    """
    device = get_device()
    
    train_loader, val_loader = get_train_dataloaders(batch_size=128)
    
    model = SmallResNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1
    )

    checkpoint_dir = "checkpoints"
    resume_path = get_latest_checkpoint(checkpoint_dir)
    
    start_epoch = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    if resume_path is not None:
        logger.info("Resuming training from checkpoint...")
        start_epoch, train_losses, train_accs, val_losses, val_accs = load_checkpoint(
            model, optimizer, resume_path, device
        )
        
    num_epochs=100
    logger.info(f"start epoch: {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "train_accs": train_accs,
                "val_losses": val_losses,
                "val_accs": val_accs,
            },
            checkpoint_dir=checkpoint_dir
        )

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    return model, train_losses, train_accs, val_losses, val_accs


if __name__ == "__main__":
    model, train_losses, train_accs, val_losses, val_accs = train()
    device = get_device()
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
