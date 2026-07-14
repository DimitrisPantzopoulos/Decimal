from .Zero import ResNet
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple

class HexDataset(Dataset):
    def __init__(self, npz_path : str) -> None:
        super().__init__()
        data : np.ndarray = np.load(npz_path)

        self.states   : torch.Tensor = torch.from_numpy(data["states"])
        self.policies : torch.Tensor = torch.from_numpy(data["policies"])
        self.values   : torch.Tensor = torch.from_numpy(data["values"])

    def __len__(self) -> int:
        return len(self.states)
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.policies[idx], self.values[idx]


def evaluate(model: ResNet, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total_policy, total_value, total = 0.0, 0.0, 0

    with torch.no_grad():
        for states, policies, values in loader:
            states   = states.to(device)
            policies = policies.to(device)
            values   = values.to(device)

            policy_logits, pred_values = model(states)

            total_policy += (-(policies * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1)).sum().item()
            total_value  += F.mse_loss(pred_values.squeeze(-1), values, reduction="sum").item()
            total        += len(states)

    return total_policy / total, total_value / total, (total_policy + total_value) / total


def train(npz_path : str, board_size : int=8, epochs: int=100, lr : float=1e-3, val_split : float=0.1) -> None:
    device : torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  : ResNet       = ResNet(board_size=board_size).to(device)
    opt    : optim.Adam   = optim.Adam(model.parameters(), lr=lr)

    dataset    = HexDataset(npz_path)
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)

    print(f"Device     : {device}")
    print(f"Dataset    : {len(dataset):,} samples  ({train_size:,} train / {val_size:,} val)")
    print(f"Board size : {board_size}x{board_size}  ({board_size * board_size} cells)")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print()

    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        for states, policies, values in train_loader:
            states          = states.to(device)
            target_policies = policies.to(device)
            target_values   = values.to(device)

            policy_logits, pred_values = model(states)

            policy_loss = -(target_policies * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
            value_loss  = F.mse_loss(pred_values.squeeze(-1), target_values)
            loss        = policy_loss + value_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_p, val_v, val_loss = evaluate(model, val_loader, device)

        epoch_bar.set_postfix(
            train = f"{train_loss:.4f}",
            val   = f"{val_loss:.4f}",
            val_p = f"{val_p:.4f}",
            val_v = f"{val_v:.4f}",
        )

    torch.save(model.state_dict(), "TrainedModels/model.pt")
    print("\nSaved model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet on Hex self-play data")
    parser.add_argument("--npz",       type=str,   required=True,  help="path to .npz dataset")
    parser.add_argument("--size",      type=int,   default=8,      help="Hex board size")
    parser.add_argument("--epochs",    type=int,   default=100,    help="number of training epochs")
    parser.add_argument("--lr",        type=float, default=1e-3,   help="learning rate")
    parser.add_argument("--val-split", type=float, default=0.1,    help="fraction of data used for validation")
    args = parser.parse_args()

    train(
        npz_path  = args.npz,
        board_size = args.size,
        epochs    = args.epochs,
        lr        = args.lr,
        val_split = args.val_split,
    )