import torch
import torch.nn as nn
import torch.optim as optim
import os

from models.cnn import SmallResNet
from dataset import get_dataloaders
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.plot_loss_epoch import plot_losses

checkpoint_dir = "checkpoints"
resume_path = "checkpoints/last.pth"

start_epoch = 0
train_losses = []
val_losses = []
best_val_loss = float("inf")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def train_one_epoch(model, loader, criterion, optimizer, device):
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
    device = get_device()
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    
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

    if os.path.exists(resume_path):
        print("Resuming training from checkpoint...")
        start_epoch, train_losses, val_losses, best_val_loss = load_checkpoint(
            model, optimizer, resume_path, device
        )

    num_epochs = 100

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
            },
            is_best=is_best,
            checkpoint_dir=checkpoint_dir
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    return model, test_loader


if __name__ == "__main__":
    model, test_loader = train()
    device = get_device()
    plot_losses(train_losses, val_losses)
