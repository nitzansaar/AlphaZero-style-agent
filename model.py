import torch
from torch import nn
import torch.nn.functional as F
from config import Config as cfg, BOARD_SIZE, ACTION_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

INPUT_PLANES = 17
POLICY_HEAD_CHANNELS = 2
VALUE_HEAD_CHANNELS = 1


class ResBlock(nn.Module):
    """Residual block with batch normalization."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class NeuralNetwork(nn.Module):
    """
    AlphaGo Zero policy/value network for the configured board size.

    The default config is 19x19 with 20 residual blocks and 256 tower channels.
    This module does not choose small-board defaults; it reads the active config
    selected by BOARD_SIZE, which defaults to 19.
    """
    def __init__(self, num_res_blocks=None, channels=None, value_hidden=None):
        super().__init__()

        # Use config values if not specified
        self.num_res_blocks = num_res_blocks if num_res_blocks is not None else cfg.NUM_RES_BLOCKS
        self.channels = channels if channels is not None else cfg.NUM_CHANNELS
        self.value_hidden = value_hidden if value_hidden is not None else cfg.VALUE_HEAD_HIDDEN
        self.board_size = BOARD_SIZE

        self.input_planes = INPUT_PLANES

        # AlphaGo Zero input: 8 history planes per player plus color-to-play.
        self.conv_init = nn.Conv2d(
            self.input_planes,
            self.channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_init = nn.BatchNorm2d(self.channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(self.channels) for _ in range(self.num_res_blocks)
        ])

        # Value head
        self.value_conv_channels = VALUE_HEAD_CHANNELS
        self.value_conv = nn.Conv2d(
            self.channels,
            self.value_conv_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.value_bn = nn.BatchNorm2d(self.value_conv_channels)
        value_fc_input = self.value_conv_channels * BOARD_SIZE * BOARD_SIZE
        self.value_fc1 = nn.Linear(value_fc_input, self.value_hidden)
        self.value_fc2 = nn.Linear(self.value_hidden, 1)

        # Policy head
        self.policy_conv_channels = POLICY_HEAD_CHANNELS
        self.policy_conv = nn.Conv2d(
            self.channels,
            self.policy_conv_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.policy_bn = nn.BatchNorm2d(self.policy_conv_channels)
        policy_fc_input = self.policy_conv_channels * BOARD_SIZE * BOARD_SIZE
        self.policy_fc = nn.Linear(policy_fc_input, ACTION_SIZE)

    def forward(self, x):
        # x shape: (batch, 17, BOARD_SIZE, BOARD_SIZE)

        # Initial convolution
        x = F.relu(self.bn_init(self.conv_init(x)))

        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_fc(p)

        return v, p

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        """Print network architecture summary."""
        print(f"Network Architecture for {self.board_size}x{self.board_size} Go")
        print(f"  Input planes: {self.input_planes}")
        print(f"  Residual blocks: {self.num_res_blocks}")
        print(f"  Channels: {self.channels}")
        print(f"  Policy head filters: {self.policy_conv_channels}")
        print(f"  Value head filters: {self.value_conv_channels}")
        print(f"  Value head hidden: {self.value_hidden}")
        print(f"  Total parameters: {self.count_parameters():,}")


if __name__ == "__main__":
    # Test network creation
    net = NeuralNetwork()
    net.print_architecture()

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    v, p = net(x)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Value output shape: {v.shape}")
    print(f"  Policy output shape: {p.shape}")
