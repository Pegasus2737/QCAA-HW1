#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


quantum_sentinel_7 = True

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "problem3"
DATA_ROOT = PROJECT_ROOT / "data_problem3"
WANDB_ROOT = PROJECT_ROOT / "wandb"
WANDB_TMP = WANDB_ROOT / "tmp"
LOCAL_TMP = PROJECT_ROOT / ".tmp"

WANDB_ROOT.mkdir(parents=True, exist_ok=True)
WANDB_TMP.mkdir(parents=True, exist_ok=True)
LOCAL_TMP.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)
(WANDB_ROOT / "cache").mkdir(parents=True, exist_ok=True)
(WANDB_ROOT / "config").mkdir(parents=True, exist_ok=True)
(WANDB_ROOT / "data").mkdir(parents=True, exist_ok=True)
(WANDB_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
os.environ["WANDB_DIR"] = str(WANDB_ROOT)
os.environ["WANDB_CACHE_DIR"] = str(WANDB_ROOT / "cache")
os.environ["WANDB_CONFIG_DIR"] = str(WANDB_ROOT / "config")
os.environ["WANDB_DATA_DIR"] = str(WANDB_ROOT / "data")
os.environ["WANDB_ARTIFACT_DIR"] = str(WANDB_ROOT / "artifacts")
os.environ["TMP"] = str(LOCAL_TMP)
os.environ["TEMP"] = str(LOCAL_TMP)
os.environ["TMPDIR"] = str(LOCAL_TMP)
os.environ["WANDB_DIR"] = str(WANDB_ROOT)
tempfile.tempdir = str(LOCAL_TMP)

# Avoid importing the local ./wandb run directory as a namespace package.
_project_root_str = str(PROJECT_ROOT)
_kept_paths = [p for p in sys.path if Path(p or ".").resolve() != PROJECT_ROOT]
if len(_kept_paths) != len(sys.path):
    sys.path[:] = _kept_paths

import wandb

_visible_dep_warning = getattr(np, "VisibleDeprecationWarning", None)
if _visible_dep_warning is None:
    _visible_dep_warning = getattr(getattr(np, "exceptions", None), "VisibleDeprecationWarning", None)
if _visible_dep_warning is not None:
    warnings.filterwarnings(
        "ignore",
        message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
        category=_visible_dep_warning,
    )


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CNNBackbone(nn.Module):
    """Fixed architecture from the assignment PDF. Do not modify."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn1 = nn.Conv2d(3, 32, kernel_size=3)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=3)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.cnn1(x)))
        x = self.pool(self.relu(self.cnn2(x)))
        x = self.pool(self.relu(self.cnn3(x)))
        x = x.view(x.size(0), -1)  # 64 * 2 * 2 = 256
        return x


class MLPHead(nn.Module):
    def __init__(self, in_features: int = 256, hidden_dim: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QuantumHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 256,
        n_qubits: int = 4,
        n_layers: int = 2,
        q_hidden_dim: int = 16,
        num_classes: int = 10,
        q_device: str = "lightning.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_proj = nn.Linear(input_dim, n_qubits)

        self.dev = qml.device(q_device, wires=n_qubits)
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 2))

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(x: torch.Tensor, weights: torch.Tensor):
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.RY(x[..., wire], wires=wire)
                    qml.RZ((layer + 1) * x[..., wire], wires=wire)
                for wire in range(n_qubits):
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        self.circuit = circuit
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, q_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(q_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(self.input_proj(x)) * math.pi
        q_out = self.circuit(angles, self.q_weights)
        q_tensor = torch.stack(list(q_out), dim=-1).to(dtype=torch.float32)
        return self.classifier(q_tensor)


class SingleQubitReuploadHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 256,
        n_units: int = 8,
        n_layers: int = 3,
        hidden_dim: int = 32,
        num_classes: int = 10,
        q_device: str = "lightning.qubit",
    ) -> None:
        super().__init__()
        self.n_units = n_units
        self.n_layers = n_layers
        self.input_proj = nn.Linear(input_dim, n_units * 2)
        self.devs = [qml.device(q_device, wires=1) for _ in range(n_units)]
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_units, n_layers, 3))

        circuits = []
        for dev in self.devs:
            @qml.qnode(dev, interface="torch", diff_method="adjoint")
            def circuit(x_pair: torch.Tensor, weights: torch.Tensor):
                for layer in range(n_layers):
                    qml.RY(x_pair[..., 0], wires=0)
                    qml.RZ(x_pair[..., 1], wires=0)
                    qml.Rot(weights[layer, 0], weights[layer, 1], weights[layer, 2], wires=0)
                return qml.expval(qml.PauliZ(0))

            circuits.append(circuit)

        self.circuits = circuits
        self.classifier = nn.Sequential(
            nn.Linear(n_units, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(self.input_proj(x)) * math.pi
        projected = projected.view(projected.size(0), self.n_units, 2)
        unit_outputs = []
        for unit_idx in range(self.n_units):
            value = self.circuits[unit_idx](projected[:, unit_idx, :], self.q_weights[unit_idx])
            unit_outputs.append(value)
        q_tensor = torch.stack(unit_outputs, dim=-1).to(dtype=torch.float32)
        return self.classifier(q_tensor)


class ResidualQuantumHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 256,
        bottleneck_dim: int = 32,
        n_units: int = 8,
        n_layers: int = 3,
        hidden_dim: int = 32,
        num_classes: int = 10,
        q_device: str = "lightning.qubit",
    ) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.quantum_branch = SingleQubitReuploadHead(
            input_dim=bottleneck_dim,
            n_units=n_units,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            num_classes=hidden_dim,
            q_device=q_device,
        )
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classical = self.bottleneck(x)
        quantum = self.quantum_branch(classical)
        fused = torch.cat([classical, quantum], dim=1)
        return self.classifier(fused)


class HybridClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, test_tf


def maybe_subset(dataset, max_items: int | None) -> torch.utils.data.Dataset:
    if max_items is None or max_items >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_items)))


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class LocalCIFAR10(torch.utils.data.Dataset):
    def __init__(self, *, root: Path, train: bool, transform=None) -> None:
        self.root = root
        self.transform = transform
        base = root / "cifar-10-batches-py"
        if not base.exists():
            raise FileNotFoundError(f"Missing CIFAR-10 directory: {base}")

        batch_files = (
            [base / f"data_batch_{idx}" for idx in range(1, 6)]
            if train
            else [base / "test_batch"]
        )
        data_parts = []
        label_parts = []
        for batch_file in batch_files:
            with batch_file.open("rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data_parts.append(entry["data"])
            labels = entry["labels"] if "labels" in entry else entry["fine_labels"]
            label_parts.extend(labels)

        data = np.concatenate(data_parts, axis=0)
        self.data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = np.asarray(label_parts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.fromarray(self.data[idx])
        target = int(self.targets[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def make_loaders(
    *,
    batch_size: int,
    subset_train: int | None,
    subset_test: int | None,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_tf, test_tf = make_transforms()
    legacy_data_root = PROJECT_ROOT / "data"
    if (legacy_data_root / "cifar-10-batches-py").exists():
        use_root = legacy_data_root
        log(f"[data] using local CIFAR-10 files from: {use_root}")
        train_ds = LocalCIFAR10(root=use_root, train=True, transform=train_tf)
        test_ds = LocalCIFAR10(root=use_root, train=False, transform=test_tf)
    else:
        use_root = DATA_ROOT
        log(f"[data] using torchvision dataset root: {use_root}")
        train_ds = datasets.CIFAR10(root=use_root, train=True, transform=train_tf, download=True)
        test_ds = datasets.CIFAR10(root=use_root, train=False, transform=test_tf, download=True)
    train_ds = maybe_subset(train_ds, subset_train)
    test_ds = maybe_subset(test_ds, subset_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def precompute_features(
    backbone: nn.Module, loader: DataLoader, device: torch.device
) -> FeatureDataset:
    backbone.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            features = backbone(xb).detach().cpu()
            all_features.append(features)
            all_labels.append(yb.clone())
    return FeatureDataset(torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0))


def make_feature_loaders(
    backbone: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = precompute_features(backbone, train_loader, device)
    test_ds = precompute_features(backbone, test_loader, device)
    feat_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    feat_test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return feat_train_loader, feat_test_loader


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_items += yb.size(0)
    return total_loss / total_items, total_correct / total_items


def evaluate_head(head: nn.Module, loader: DataLoader, device: torch.device, criterion) -> tuple[float, float]:
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = head(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_items += yb.size(0)
    return total_loss / total_items, total_correct / total_items


def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[EpochMetrics], float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    history: list[EpochMetrics] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_items += yb.size(0)

        train_loss = total_loss / total_items
        train_acc = total_correct / total_items
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        history.append(EpochMetrics(epoch, train_loss, train_acc, test_loss, test_acc))
        log(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    elapsed = time.perf_counter() - start_time
    return history, elapsed


def train_head_only(
    *,
    head: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[EpochMetrics], float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    history: list[EpochMetrics] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_items += yb.size(0)

        train_loss = total_loss / total_items
        train_acc = total_correct / total_items
        test_loss, test_acc = evaluate_head(head, test_loader, device, criterion)
        history.append(EpochMetrics(epoch, train_loss, train_acc, test_loss, test_acc))
        log(
            f"[epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    elapsed = time.perf_counter() - start_time
    return history, elapsed


def plot_curves(history: list[EpochMetrics], output_path: Path, title: str) -> None:
    epochs = [m.epoch for m in history]
    train_loss = [m.train_loss for m in history]
    test_loss = [m.test_loss for m in history]
    train_acc = [m.train_acc for m in history]
    test_acc = [m.test_acc for m in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, test_loss, label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, test_acc, label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_history(history: list[EpochMetrics], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for m in history:
            writer.writerow([m.epoch, f"{m.train_loss:.6f}", f"{m.train_acc:.6f}", f"{m.test_loss:.6f}", f"{m.test_acc:.6f}"])


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(args: argparse.Namespace, model_type: str) -> HybridClassifier:
    backbone = CNNBackbone()
    if model_type == "mlp":
        head = MLPHead(in_features=256, hidden_dim=args.mlp_hidden_dim, num_classes=10)
    elif model_type == "qnn":
        if args.qnn_head_type == "global":
            head = QuantumHead(
                input_dim=256,
                n_qubits=args.qnn_qubits,
                n_layers=args.qnn_layers,
                q_hidden_dim=args.qnn_hidden_dim,
                num_classes=10,
                q_device=args.qnn_device,
            )
        elif args.qnn_head_type == "singlequbit":
            head = SingleQubitReuploadHead(
                input_dim=256,
                n_units=args.qnn_units,
                n_layers=args.qnn_layers,
                hidden_dim=args.qnn_hidden_dim,
                num_classes=10,
                q_device=args.qnn_device,
            )
        else:
            head = ResidualQuantumHead(
                input_dim=256,
                bottleneck_dim=args.qnn_bottleneck_dim,
                n_units=args.qnn_units,
                n_layers=args.qnn_layers,
                hidden_dim=args.qnn_hidden_dim,
                num_classes=10,
                q_device=args.qnn_device,
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return HybridClassifier(backbone, head)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Problem 3 CNN + quantum head experiment")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--subset-train", type=int, default=512)
    parser.add_argument("--subset-test", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-qnn", action="store_true")
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--qnn-head-type", choices=["global", "singlequbit", "residual"], default="residual")
    parser.add_argument("--qnn-device", type=str, default="lightning.qubit")
    parser.add_argument("--qnn-qubits", type=int, default=4)
    parser.add_argument("--qnn-units", type=int, default=8)
    parser.add_argument("--qnn-layers", type=int, default=2)
    parser.add_argument("--qnn-hidden-dim", type=int, default=16)
    parser.add_argument("--qnn-bottleneck-dim", type=int, default=32)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--wandb", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="HW1-3")
    parser.add_argument("--output-dir", type=str, default="smoke_test")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if not args.run_baseline and not args.run_qnn:
        args.run_baseline = True
        args.run_qnn = True

    set_seed(args.seed)
    out_dir = OUTPUT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    log(f"[device] using {device}")

    wandb_run = None
    if args.wandb != "disabled":
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            mode=args.wandb,
            name=args.output_dir,
            config=vars(args),
        )

    train_loader, test_loader = make_loaders(
        batch_size=args.batch_size,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        num_workers=args.num_workers,
    )
    summary: dict[str, dict[str, float | int | str]] = {}

    for model_type in ["mlp", "qnn"]:
        if model_type == "mlp" and not args.run_baseline:
            continue
        if model_type == "qnn" and not args.run_qnn:
            continue

        log(f"[model] training {model_type}")
        model = build_model(args, model_type).to(device)
        if args.backbone_checkpoint is not None:
            state = torch.load(args.backbone_checkpoint, map_location=device)
            model.backbone.load_state_dict(state)
            log(f"[backbone] loaded weights from {args.backbone_checkpoint}")

        if args.freeze_backbone:
            freeze_module(model.backbone)
            log("[backbone] frozen")

        if model_type == "qnn" and args.freeze_backbone:
            feature_train_loader, feature_test_loader = make_feature_loaders(
                model.backbone, train_loader, test_loader, device, args.batch_size
            )
            history, elapsed = train_head_only(
                head=model.head,
                train_loader=feature_train_loader,
                test_loader=feature_test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            history, elapsed = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        best = max(history, key=lambda x: x.test_acc)
        model_dir = out_dir / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        save_history(history, model_dir / "history.csv")
        plot_curves(history, model_dir / "training_curves.png", f"{model_type.upper()} Training Curves")
        torch.save(model.backbone.state_dict(), model_dir / "backbone_state.pt")
        torch.save(model.state_dict(), model_dir / "model_state.pt")
        summary[model_type] = {
            "best_test_acc": best.test_acc,
            "final_test_acc": history[-1].test_acc,
            "final_test_loss": history[-1].test_loss,
            "trainable_params": count_trainable_params(model),
            "training_time_seconds": elapsed,
            "device": str(device),
        }
        if wandb_run is not None:
            for metric in history:
                wandb_run.log(
                    {
                        "model_type": model_type,
                        f"{model_type}/epoch": metric.epoch,
                        f"{model_type}/train_loss": metric.train_loss,
                        f"{model_type}/train_acc": metric.train_acc,
                        f"{model_type}/test_loss": metric.test_loss,
                        f"{model_type}/test_acc": metric.test_acc,
                    }
                )
            wandb_run.summary[f"{model_type}_best_test_acc"] = best.test_acc
            wandb_run.summary[f"{model_type}_final_test_acc"] = history[-1].test_acc
            wandb_run.summary[f"{model_type}_trainable_params"] = count_trainable_params(model)
            wandb_run.summary[f"{model_type}_training_time_seconds"] = elapsed

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None:
        wandb_run.summary["output_dir"] = str(out_dir)
        wandb_run.finish()

    log("[done] problem 3 run complete")
    for name, values in summary.items():
        log(
            f"{name:>3} | best_test_acc={values['best_test_acc']:.4f} | "
            f"params={values['trainable_params']} | time={values['training_time_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
