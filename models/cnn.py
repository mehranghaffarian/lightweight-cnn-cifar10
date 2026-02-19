import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A single residual block with optional downsampling and dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the first convolution. Default: 1.
        dropout (float, optional): Dropout probability after first conv. Default: 0.0.

    Notes:
        - If stride != 1 or in_channels != out_channels, a 1x1 convolution
          is applied in the shortcut to match dimensions.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(p=dropout)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output feature map of same or downsampled spatial size
                          depending on stride.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class SmallResNet(nn.Module):
    """
    A small ResNet-inspired CNN for CIFAR-10 classification (<400k parameters).

    Architecture:
        - Stem: Conv3x3 -> BN -> ReLU
        - Layer1: 2 x ResidualBlock(32,32)
        - Layer2: ResidualBlock(32,64,stride=2,dropout=0.2) + ResidualBlock(64,64)
        - Layer3: ResidualBlock(64,112,stride=2,dropout=0.3)
        - AdaptiveAvgPool -> Flatten -> Fully-connected layer

    Args:
        num_classes (int): Number of output classes (default=10 for CIFAR-10)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, dropout=0.2),
            ResidualBlock(64, 64)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(64, 112, stride=2, dropout=0.3)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(112, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input image batch, shape (B, 3, 32, 32)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = SmallResNet()
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")