"""Small MLPs used as both teacher and student, and a small MNIST MLP."""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}[name]


class MLP(nn.Module):
    """Plain feed-forward MLP. Also used as the fixed random *teacher*."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        hidden_depth: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(prev, hidden_width))
            layers.append(_act(activation))
            prev = hidden_width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[off:off + n].view_as(p))
        off += n


def flat_grad(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    return torch.cat(grads)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class SmallCNN(nn.Module):
    """Compact CIFAR-scale CNN: 2x conv block + FC. Deterministic (no dropout)."""

    def __init__(self, in_channels: int = 3, base: int = 32, fc: int = 128,
                 n_classes: int = 10, input_hw: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, 2 * base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * base, 2 * base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        h_out = input_hw // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((2 * base) * h_out * h_out, fc),
            nn.ReLU(inplace=True),
            nn.Linear(fc, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MultiHeadMLP(nn.Module):
    """Shared MLP backbone + per-task 2-way heads (task-incremental protocol).

    Forward requires a task_id. The forward returns logits from head[task_id];
    train and eval always route through the correct head.
    """

    def __init__(self, input_dim: int, hidden: int, depth: int,
                 n_tasks: int, n_classes_per_task: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU(inplace=True))
            prev = hidden
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, n_classes_per_task) for _ in range(n_tasks)]
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.heads[task_id](self.backbone(x))


class MultiHeadCNN(nn.Module):
    """Shared SmallCNN backbone + per-task 2-way heads (TIL)."""

    def __init__(self, in_channels: int = 3, base: int = 32, fc: int = 128,
                 n_tasks: int = 5, n_classes_per_task: int = 2,
                 input_hw: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, 2 * base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * base, 2 * base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        h_out = input_hw // 4
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear((2 * base) * h_out * h_out, fc),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(fc, n_classes_per_task) for _ in range(n_tasks)]
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.heads[task_id](self.trunk(self.features(x)))
