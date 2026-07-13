from ..Board import HexBoard
from typing import Tuple

import torch.nn.functional as F
import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, num_hidden : int=4) -> None:
        super().__init__()
        self.conv1 : nn.Conv2d = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.conv2 : nn.Conv2d = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)

        self.bn1 : nn.BatchNorm2d = nn.BatchNorm2d(num_hidden)
        self.bn2 : nn.BatchNorm2d = nn.BatchNorm2d(num_hidden)

    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        residual : torch.Tensor = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)) + residual)
        return x 
    
class ResNet(nn.Module):
    def __init__(
            self,
            num_cells       : int = 64,
            in_channels     : int = 3,
            num_hidden      : int = 64,
            num_res_blocks  : int = 3,
            policy_head_out : int = 32,
            value_head_out  : int = 32,
            
        ) -> None:
        super().__init__()

        self.start_block : nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backbone : nn.ModuleList = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )

        self.policy_head : nn.Sequential = nn.Sequential(
            nn.Conv2d(num_hidden, policy_head_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(policy_head_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_head_out * num_cells, num_cells)
        )

        self.value_head : nn.Sequential = nn.Sequential(
            nn.Conv2d(num_hidden, value_head_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(value_head_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_head_out * num_cells, 1),
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
    resnet : ResNet   = ResNet(num_cells=board.num_cells)

    encoded_board : torch.Tensor = torch.from_numpy(board.encode_board().copy()).unsqueeze(0)

    policy, value = resnet(encoded_board)

    print(f"Policy: \n {policy}\n")
    print(f"value: \n {value} \n")

if __name__ == "__main__":
    test_compatibility()