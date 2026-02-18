import torch
from models.cnn import SmallResNet
from dataset import *
import matplotlib.pyplot as plt
import torchvision

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


def show_predictions(model, test_loader, device, num_images=8):
    classes = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )
    
    model.eval()

    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        img = images[i]

        # unnormalize if needed
        img = img.permute(1, 2, 0)

        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])

        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        color = "green" if preds[i] == labels[i] else "red"

        axes[i].imshow(img)
        axes[i].set_title(
            f"T:{classes[labels[i]]}\nP:{classes[preds[i]]}",
            color=color
        )
        axes[i].axis("off")

    plt.show()


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
    show_predictions(model, test_loader, device)

