from ..Board import HexBoard
from typing import Tuple

import torch.nn.functional as F
import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, num_hidden : int=4) -> None:
        super().__init__()
        self.conv1 : nn.Conv2d = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)
        self.conv2 : nn.Conv2d = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1, bias=False)

        self.bn1 : nn.BatchNorm2d = nn.BatchNorm2d(num_hidden)
        self.bn2 : nn.BatchNorm2d = nn.BatchNorm2d(num_hidden)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual : torch.Tensor = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + residual)
        return x 
    
class ResNet(nn.Module):
    def __init__(
            self,
            board_size           : int = 11,
            in_channels          : int = 3,
            num_hidden           : int = 64,
            num_res_blocks       : int = 4,
            policy_head_channels : int = 4,
            value_head_channels  : int = 2,
            value_hidden         : int = 128,
            
        ) -> None:
        super().__init__()

        self.num_cells = board_size * board_size

        self.start_block : nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backbone : nn.Sequential = nn.Sequential(
            *[ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )

        self.policy_head : nn.Sequential = nn.Sequential(
            nn.Conv2d(num_hidden, policy_head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(policy_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_head_channels * self.num_cells, self.num_cells)
        )

        self.value_head : nn.Sequential = nn.Sequential(
            nn.Conv2d(num_hidden, value_head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(value_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_head_channels * self.num_cells, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh()
        )

    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.start_block(x)

        for res_block in self.backbone:
            x = res_block(x)
        
        policy : torch.Tensor = self.policy_head(x)
        value  : torch.Tensor = self.value_head(x)
        return policy, value

def test_compatibility() -> None:
    BOARD_SIZE : int = 8

    board  : HexBoard = HexBoard(size=BOARD_SIZE)
    resnet : ResNet   = ResNet(board_size=BOARD_SIZE)

    encoded_board : torch.Tensor = torch.from_numpy(board.encode_board().copy()).unsqueeze(0)

    policy, value = resnet(encoded_board)

    print(f"Policy: \n {policy}\n")
    print(f"value: \n {value} \n")

if __name__ == "__main__":
    test_compatibility()