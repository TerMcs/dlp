from torch import nn
import torch.nn.functional as f

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(channels[2])
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[2],
                kernel_size=1,
                stride=2,
                padding=0
            ),
            nn.BatchNorm2d(channels[2])
        )

    def forward(self, x):
        shortcut = x

        residual_block = self.residual_block(x)
        shortcut = self.shortcut(x)
        x = f.relu(residual_block + shortcut)
        return x

class ResNetBasic(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.residual_block_1 = ResidualBlock(channels=[1, 4, 8])
        self.residual_block_2 = ResidualBlock(channels=[8, 16, 32])

        self.linear = nn.Linear(7*7*32, num_classes) # stride is 2 and block is applied twice so dimensions go from 28*28 to 14*14 to 7*7

    def forward(self, x):
        out = self.residual_block_1(x)
        out = self.residual_block_2(out)

        output = self.linear(out.view(-1, 7*7*32)) # Can't figure out how to make nn.Flatten work with this.
        return output