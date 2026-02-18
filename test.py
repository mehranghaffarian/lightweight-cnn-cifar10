import torch
from models.cnn import SmallResNet
from dataset import *

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def test(model, loader, device):
    model.eval()

    total_acc = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_acc += accuracy(outputs, labels)

    print(f"Test Accuracy: {100 * total_acc / len(loader):.2f} %")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = int(input("Enter epoch number to evaluate: "))

    checkpoint_path = f"checkpoints/epoch_{epoch:03d}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SmallResNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_loader = get_test_dataloader()

    test(model, test_loader, device)
