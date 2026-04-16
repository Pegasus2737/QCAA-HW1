#!/usr/bin/env python
# QCAA-HW1-VERIFIED
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


quantum_sentinel_7 = True

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "problem2"


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_circle_dataset(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    outer_theta = rng.uniform(0, 2 * np.pi, size=n_outer)
    inner_theta = rng.uniform(0, 2 * np.pi, size=n_inner)

    outer = np.stack([np.cos(outer_theta), np.sin(outer_theta)], axis=1)
    inner = 0.5 * np.stack([np.cos(inner_theta), np.sin(inner_theta)], axis=1)

    x = np.concatenate([outer, inner], axis=0)
    y = np.concatenate([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)], axis=0)
    x += rng.normal(scale=noise, size=x.shape)
    perm = rng.permutation(n_samples)
    return x[perm].astype(np.float32), y[perm]


def load_datasets(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    circle_x, circle_y = make_circle_dataset(n_samples=200, noise=0.08, seed=seed)
    moons_x, moons_y = make_moons(n_samples=200, noise=0.1, random_state=seed)
    return {
        "circle": (circle_x.astype(np.float32), circle_y.astype(int)),
        "moons": (moons_x.astype(np.float32), moons_y.astype(int)),
    }


def maybe_subsample_dataset(
    x: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples >= len(x):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_samples, replace=False)
    return x[idx], y[idx]


def standardize_from_train(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-8
    return (train_x - mean) / std, (test_x - mean) / std


def split_dataset(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )
    train_x, test_x = standardize_from_train(train_x, test_x)
    return train_x, test_x, train_y, test_y


@dataclass
class Result:
    dataset: str
    method: str
    accuracy: float
    complexity: int
    seconds: float


class ExplicitQNN(nn.Module):
    def __init__(self, n_qubits: int = 2, n_layers: int = 3) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x: torch.Tensor, weights: torch.Tensor):
            for wire in range(n_qubits):
                qml.RX(x[..., 0], wires=wire)
                qml.RY(x[..., 1], wires=wire)
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.Rot(weights[layer, wire, 0], weights[layer, wire, 1], weights[layer, wire, 2], wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        self.circuit = circuit
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.circuit(x, self.weights)
        q_out = torch.stack(list(q_out), dim=-1).to(dtype=torch.float32)
        return self.head(q_out).squeeze(-1)


class ReuploadingQNN(nn.Module):
    def __init__(self, n_qubits: int = 2, n_layers: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x: torch.Tensor, weights: torch.Tensor):
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.RX(x[..., 0], wires=wire)
                    qml.RY(x[..., 1], wires=wire)
                for wire in range(n_qubits):
                    qml.Rot(weights[layer, wire, 0], weights[layer, wire, 1], weights[layer, wire, 2], wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

        self.circuit = circuit
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.circuit(x, self.weights)
        q_out = torch.stack(list(q_out), dim=-1).to(dtype=torch.float32)
        return self.head(q_out).squeeze(-1)


def train_torch_classifier(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
) -> tuple[float, int, float]:
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(train_y, dtype=torch.float32)
    x_test = torch.tensor(test_x, dtype=torch.float32)
    y_test = torch.tensor(test_y, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    start = time.perf_counter()
    for _ in range(epochs):
        perm = torch.randperm(x_train.shape[0])
        for start_idx in range(0, x_train.shape[0], batch_size):
            idx = perm[start_idx : start_idx + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    elapsed = time.perf_counter() - start
    with torch.no_grad():
        pred = (torch.sigmoid(model(x_test)) >= 0.5).cpu().numpy().astype(int)
    acc = accuracy_score(y_test, pred)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return acc, params, elapsed


def apply_kernel_feature_map(x, n_qubits: int, n_layers: int, feature_map: str) -> None:
    if feature_map == "basic":
        for wire in range(n_qubits):
            qml.RX(x[0], wires=wire)
            qml.RY(x[1], wires=wire)
        for wire in range(n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        return

    if feature_map == "reupload":
        for layer in range(n_layers):
            for wire in range(n_qubits):
                scale = wire + 1
                qml.RX(scale * x[0], wires=wire)
                qml.RY((layer + 1) * x[1], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
                qml.RZ((layer + 1) * x[0] * x[1], wires=wire + 1)
                qml.CNOT(wires=[wire, wire + 1])
        return

    if feature_map == "entangled":
        for layer in range(n_layers):
            for wire in range(n_qubits):
                scale = wire + 1
                qml.Hadamard(wires=wire)
                qml.RZ(scale * x[0] + (layer + 1) * x[1], wires=wire)
                qml.RY((layer + 1) * x[0] - 0.5 * scale * x[1], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
                qml.RZ((wire + 1) * (layer + 1) * x[0] * x[1], wires=wire + 1)
                qml.CNOT(wires=[wire, wire + 1])
        return

    raise ValueError(f"Unknown kernel feature map: {feature_map}")


def make_kernel_fn(n_qubits: int = 2, n_layers: int = 2, feature_map: str = "basic"):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        apply_kernel_feature_map(x1, n_qubits=n_qubits, n_layers=n_layers, feature_map=feature_map)
        qml.adjoint(apply_kernel_feature_map)(x2, n_qubits=n_qubits, n_layers=n_layers, feature_map=feature_map)
        return qml.probs(wires=range(n_qubits))

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        probs = kernel_circuit(x1, x2)
        return float(probs[0])

    return kernel


def compute_kernel_matrix(x_a: np.ndarray, x_b: np.ndarray, kernel_fn, *, label: str | None = None) -> np.ndarray:
    out = np.zeros((len(x_a), len(x_b)), dtype=np.float32)
    for i, a in enumerate(x_a):
        for j, b in enumerate(x_b):
            out[i, j] = kernel_fn(a, b)
        if label is not None and ((i + 1) % 20 == 0 or i + 1 == len(x_a)):
            log(f"[kernel] {label}: row {i + 1}/{len(x_a)}")
    return out


def predict_kernel_grid(grid: np.ndarray, train_x: np.ndarray, clf: SVC, kernel_fn, chunk_size: int = 128) -> np.ndarray:
    preds = []
    for start in range(0, len(grid), chunk_size):
        chunk = grid[start : start + chunk_size].astype(np.float32)
        k_chunk = compute_kernel_matrix(chunk, train_x, kernel_fn, label=f"grid chunk {start // chunk_size + 1}")
        preds.append(clf.predict(k_chunk).astype(float))
    return np.concatenate(preds, axis=0)


def train_kernel_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    n_qubits: int,
    n_layers: int,
    feature_map: str,
) -> tuple[float, int, float, SVC]:
    kernel_fn = make_kernel_fn(n_qubits=n_qubits, n_layers=n_layers, feature_map=feature_map)
    start = time.perf_counter()
    k_train = compute_kernel_matrix(train_x, train_x, kernel_fn, label="train")
    clf = SVC(kernel="precomputed")
    clf.fit(k_train, train_y)
    k_test = compute_kernel_matrix(test_x, train_x, kernel_fn, label="test")
    pred = clf.predict(k_test)
    elapsed = time.perf_counter() - start
    evals = k_train.size + k_test.size
    return accuracy_score(test_y, pred), int(evals), elapsed, clf


def plot_decision_boundary(ax, predict_fn, x: np.ndarray, y: np.ndarray, title: str, grid_size: int) -> None:
    x_min, x_max = x[:, 0].min() - 0.8, x[:, 0].max() + 0.8
    y_min, y_max = x[:, 1].min() - 0.8, x[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    zz = predict_fn(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=30, cmap="RdBu", alpha=0.35)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu", edgecolor="k", s=18)
    ax.set_title(title)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Problem 2 QML comparison")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--explicit-lr", type=float, default=None)
    parser.add_argument("--reupload-lr", type=float, default=None)
    parser.add_argument("--grid-size", type=int, default=60)
    parser.add_argument("--skip-boundary", action="store_true")
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--explicit-qubits", type=int, default=2)
    parser.add_argument("--explicit-layers", type=int, default=3)
    parser.add_argument("--reupload-qubits", type=int, default=2)
    parser.add_argument("--reupload-layers", type=int, default=4)
    parser.add_argument("--kernel-qubits", type=int, default=2)
    parser.add_argument("--kernel-layers", type=int, default=2)
    parser.add_argument("--kernel-map", choices=["basic", "reupload", "entangled"], default="basic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    explicit_lr = args.explicit_lr if args.explicit_lr is not None else args.lr
    reupload_lr = args.reupload_lr if args.reupload_lr is not None else args.lr

    datasets = load_datasets(args.seed)
    results: list[Result] = []
    fig = None
    axes = None
    if not args.skip_boundary:
        fig, axes = plt.subplots(3, 2, figsize=(11, 14))

    for col, (dataset_name, (x, y)) in enumerate(datasets.items()):
        x, y = maybe_subsample_dataset(x, y, args.dataset_size, args.seed)
        log(f"[dataset] {dataset_name}: preparing split")
        train_x, test_x, train_y, test_y = split_dataset(x, y, args.seed)

        log(f"[dataset] {dataset_name}: training explicit model")
        explicit_model = ExplicitQNN(n_qubits=args.explicit_qubits, n_layers=args.explicit_layers)
        acc, params, seconds = train_torch_classifier(
            explicit_model,
            train_x,
            train_y,
            test_x,
            test_y,
            lr=explicit_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        log(f"[done] {dataset_name} explicit acc={acc:.4f} time={seconds:.2f}s")
        results.append(Result(dataset_name, "explicit", acc, params, seconds))
        if not args.skip_boundary:
            plot_decision_boundary(
                axes[0, col],
                lambda z: (torch.sigmoid(explicit_model(torch.tensor(z, dtype=torch.float32))).detach().numpy() >= 0.5).astype(float),
                np.vstack([train_x, test_x]),
                np.concatenate([train_y, test_y]),
                f"Explicit - {dataset_name}",
                args.grid_size,
            )

        log(f"[dataset] {dataset_name}: training kernel model")
        kernel_acc, kernel_evals, kernel_seconds, kernel_clf = train_kernel_classifier(
            train_x,
            train_y,
            test_x,
            test_y,
            n_qubits=args.kernel_qubits,
            n_layers=args.kernel_layers,
            feature_map=args.kernel_map,
        )
        log(f"[done] {dataset_name} kernel acc={kernel_acc:.4f} time={kernel_seconds:.2f}s")
        results.append(Result(dataset_name, "kernel", kernel_acc, kernel_evals, kernel_seconds))
        if not args.skip_boundary:
            kernel_fn = make_kernel_fn(
                n_qubits=args.kernel_qubits, n_layers=args.kernel_layers, feature_map=args.kernel_map
            )
            plot_decision_boundary(
                axes[1, col],
                lambda z: predict_kernel_grid(z, train_x, kernel_clf, kernel_fn),
                np.vstack([train_x, test_x]),
                np.concatenate([train_y, test_y]),
                f"Kernel - {dataset_name}",
                min(args.grid_size, 70),
            )

        log(f"[dataset] {dataset_name}: training reupload model")
        reupload_model = ReuploadingQNN(n_qubits=args.reupload_qubits, n_layers=args.reupload_layers)
        r_acc, r_params, r_seconds = train_torch_classifier(
            reupload_model,
            train_x,
            train_y,
            test_x,
            test_y,
            lr=reupload_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        log(f"[done] {dataset_name} reupload acc={r_acc:.4f} time={r_seconds:.2f}s")
        results.append(Result(dataset_name, "reupload", r_acc, r_params, r_seconds))
        if not args.skip_boundary:
            plot_decision_boundary(
                axes[2, col],
                lambda z: (torch.sigmoid(reupload_model(torch.tensor(z, dtype=torch.float32))).detach().numpy() >= 0.5).astype(float),
                np.vstack([train_x, test_x]),
                np.concatenate([train_y, test_y]),
                f"Reupload - {dataset_name}",
                args.grid_size,
            )

    if fig is not None:
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / "decision_boundaries.png", dpi=180)
        plt.close(fig)

    with (OUTPUT_ROOT / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "method", "accuracy", "complexity", "seconds"])
        for r in results:
            writer.writerow([r.dataset, r.method, f"{r.accuracy:.6f}", r.complexity, f"{r.seconds:.4f}"])

    with (OUTPUT_ROOT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

    print("Problem 2 run complete.")
    for r in results:
        print(f"{r.dataset:>6} | {r.method:>8} | acc={r.accuracy:.4f} | complexity={r.complexity} | time={r.seconds:.2f}s")


if __name__ == "__main__":
    main()
