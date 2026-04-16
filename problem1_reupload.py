#!/usr/bin/env python
# QCAA-HW1-VERIFIED
from __future__ import annotations

import argparse
import csv

quantum_sentinel_7 = True
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "problem1"
WANDB_ROOT = PROJECT_ROOT / "wandb"


def ensure_runtime_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    WANDB_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(WANDB_ROOT))
    os.environ.setdefault("WANDB_CACHE_DIR", str(WANDB_ROOT / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(WANDB_ROOT / "config"))
    os.environ.setdefault("TMP", str(PROJECT_ROOT / ".tmp"))
    os.environ.setdefault("TEMP", str(PROJECT_ROOT / ".tmp"))
    Path(os.environ["TMP"]).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.exp(x[:, 0]) + x[:, 1])


def sample_uniform(n_samples: int, ranges: np.ndarray) -> torch.Tensor:
    out = torch.zeros(n_samples, ranges.shape[0], dtype=torch.float32)
    for i in range(ranges.shape[0]):
        low, high = ranges[i]
        out[:, i] = torch.rand(n_samples, dtype=torch.float32) * (high - low) + low
    return out


def build_dataset(n_train: int, n_test: int) -> tuple[torch.Tensor, ...]:
    train_ranges = np.array([0.0, 0.5] * 2).reshape(2, 2)
    test_ranges = np.array([0.5, 1.0] * 2).reshape(2, 2)
    train_x = sample_uniform(n_train, train_ranges)
    test_x = sample_uniform(n_test, test_ranges)
    train_y = target_function(train_x)
    test_y = target_function(test_x)
    return train_x, train_y, test_x, test_y


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in iterate_minibatches(x, batch_size=batch_size, shuffle=False):
            preds.append(model(xb))
    pred = torch.cat(preds, dim=0)
    return torch.mean((pred - y) ** 2).item()


def iterate_minibatches(
    x: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    batch_size: int,
    shuffle: bool,
) -> Iterable[tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
    n = x.shape[0]
    indices = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        if y is None:
            yield x[batch_idx]
        else:
            yield x[batch_idx], y[batch_idx]


class ReuploadingRegressor(nn.Module):
    def __init__(
        self,
        *,
        n_qubits: int,
        n_layers: int,
        device_name: str,
        encoding: str,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding = encoding
        self.device_name = device_name
        self.dev = qml.device(device_name, wires=n_qubits)
        weight_shape = (n_layers, n_qubits, 3)
        self.weights = nn.Parameter(0.01 * torch.randn(weight_shape, dtype=torch.float32))
        self.readout = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        diff_method = "backprop" if device_name == "default.qubit" else "adjoint"

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def circuit(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    f0 = features[..., wire % 2]
                    f1 = features[..., (wire + 1) % 2]
                    if encoding == "rx_ry":
                        qml.RX(f0, wires=wire)
                        qml.RY(f1, wires=wire)
                    elif encoding == "ry_rz":
                        qml.RY(f0, wires=wire)
                        qml.RZ(f1, wires=wire)
                    else:
                        raise ValueError(f"Unsupported encoding: {encoding}")
                for wire in range(n_qubits):
                    qml.Rot(weights[layer, wire, 0], weights[layer, wire, 1], weights[layer, wire, 2], wires=wire)
                if n_qubits > 1:
                    for wire in range(n_qubits):
                        qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits))

        self.circuit = circuit
        self.post = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if self.device_name == "default.qubit":
            q_out = self.circuit(x, self.weights)
            if isinstance(q_out, (tuple, list)):
                stacked = torch.stack(list(q_out), dim=-1)
            else:
                stacked = q_out
        else:
            outputs = []
            for sample in x:
                q_out = self.circuit(sample, self.weights)
                if isinstance(q_out, tuple):
                    q_out = torch.stack(list(q_out))
                outputs.append(q_out)
            stacked = torch.stack(outputs, dim=0)

        stacked = stacked.to(dtype=torch.float32)
        pred = self.post(stacked).squeeze(-1)
        return self.readout * pred + self.bias


@dataclass
class RunConfig:
    name: str
    n_qubits: int
    n_layers: int
    encoding: str
    group: str = "custom"
    lr: float = 0.03
    epochs: int = 80
    batch_size: int = 32
    eval_every: int = 5


CONFIG_LIBRARY = [
    RunConfig(name="q2_l2_rxry", n_qubits=2, n_layers=2, encoding="rx_ry", group="quick"),
    RunConfig(name="q2_l4_rxry", n_qubits=2, n_layers=4, encoding="rx_ry", group="quick"),
    RunConfig(name="q2_l6_rxry", n_qubits=2, n_layers=6, encoding="rx_ry", group="quick"),
    RunConfig(name="q3_l4_ryrz", n_qubits=3, n_layers=4, encoding="ry_rz", group="control"),
    RunConfig(name="q3_l2_ryrz", n_qubits=3, n_layers=2, encoding="ry_rz", group="layers"),
    RunConfig(name="q3_l6_ryrz", n_qubits=3, n_layers=6, encoding="ry_rz", group="layers"),
    RunConfig(name="q3_l8_ryrz", n_qubits=3, n_layers=8, encoding="ry_rz", group="layers"),
    RunConfig(name="q2_l4_ryrz", n_qubits=2, n_layers=4, encoding="ry_rz", group="qubits"),
    RunConfig(name="q4_l4_ryrz", n_qubits=4, n_layers=4, encoding="ry_rz", group="qubits"),
    RunConfig(name="q3_l4_rxry", n_qubits=3, n_layers=4, encoding="rx_ry", group="encoding"),
]

PRESET_CONFIG_NAMES = {
    "quick": ["q2_l2_rxry", "q2_l4_rxry", "q2_l6_rxry", "q3_l4_ryrz"],
    "formal": [
        "q3_l2_ryrz",
        "q3_l4_ryrz",
        "q3_l6_ryrz",
        "q3_l8_ryrz",
        "q2_l4_ryrz",
        "q4_l4_ryrz",
        "q3_l4_rxry",
    ],
}

CONFIG_MAP = {cfg.name: cfg for cfg in CONFIG_LIBRARY}


def build_grid_configs(
    *,
    qubits: list[int],
    layers: list[int],
    encodings: list[str],
    lrs: list[float],
    epochs: int,
    batch_size: int,
    eval_every: int,
) -> list[RunConfig]:
    configs = []
    for n_qubits, n_layers, encoding, lr in itertools.product(qubits, layers, encodings, lrs):
        lr_tag = str(lr).replace(".", "p")
        name = f"grid_q{n_qubits}_l{n_layers}_{encoding}_lr{lr_tag}"
        configs.append(
            RunConfig(
                name=name,
                n_qubits=n_qubits,
                n_layers=n_layers,
                encoding=encoding,
                group="grid",
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                eval_every=eval_every,
            )
        )
    return configs


def maybe_init_wandb(
    mode: str,
    config: RunConfig,
    seed: int,
    *,
    entity: str | None,
    project: str,
):
    if mode == "disabled":
        return None
    import wandb

    return wandb.init(
        entity=entity,
        project=project,
        dir=str(WANDB_ROOT),
        mode=mode,
        config={"seed": seed, **asdict(config)},
        name=config.name,
        reinit=True,
    )


def maybe_log_wandb_artifact(run, *, training_curve: Path | None = None, fourier_plot: Path | None = None) -> None:
    if run is None:
        return
    import wandb

    payload = {}
    if training_curve is not None and training_curve.exists():
        payload["training_curves"] = wandb.Image(str(training_curve))
    if fourier_plot is not None and fourier_plot.exists():
        payload["fourier_spectra"] = wandb.Image(str(fourier_plot))
    if payload:
        run.log(payload)


def maybe_log_wandb_summary(
    *,
    mode: str,
    entity: str | None,
    project: str,
    seed: int,
    best_name: str,
    best_test_mse: float,
    training_curve: Path | None,
    fourier_plot: Path | None,
) -> None:
    if mode == "disabled":
        return
    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        dir=str(WANDB_ROOT),
        mode=mode,
        name=f"{best_name}_summary",
        job_type="summary",
        config={"seed": seed, "best_name": best_name, "best_test_mse": best_test_mse},
        reinit=True,
    )
    run.summary["best_test_mse"] = best_test_mse
    maybe_log_wandb_artifact(run, training_curve=training_curve, fourier_plot=fourier_plot)
    run.finish()


def train_single_run(
    *,
    config: RunConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    device_name: str,
    wandb_mode: str,
    wandb_entity: str | None,
    wandb_project: str,
) -> dict:
    set_seed(seed)
    model = ReuploadingRegressor(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        device_name=device_name,
        encoding=config.encoding,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    history: list[dict] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_test_mse = math.inf
    start_time = time.perf_counter()
    run = maybe_init_wandb(
        wandb_mode,
        config,
        seed,
        entity=wandb_entity,
        project=wandb_project,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in iterate_minibatches(train_x, train_y, batch_size=config.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        mean_batch_loss = float(np.mean(batch_losses))
        should_eval = epoch == 1 or epoch == config.epochs or epoch % config.eval_every == 0
        if should_eval:
            train_mse = compute_mse(model, train_x, train_y, batch_size=config.batch_size)
            test_mse = compute_mse(model, test_x, test_y, batch_size=config.batch_size)
            history.append(
                {
                    "epoch": epoch,
                    "batch_loss": mean_batch_loss,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                }
            )
            if test_mse < best_test_mse:
                best_test_mse = test_mse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if run is not None:
                run.log({"epoch": epoch, "train_mse": train_mse, "test_mse": test_mse, "batch_loss": mean_batch_loss})
        elif run is not None:
            run.log({"epoch": epoch, "batch_loss": mean_batch_loss})

    elapsed = time.perf_counter() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    if run is not None:
        run.summary["best_test_mse"] = best_test_mse
        run.finish()

    return {
        "model": model,
        "history": history,
        "train_mse": compute_mse(model, train_x, train_y, batch_size=config.batch_size),
        "test_mse": compute_mse(model, test_x, test_y, batch_size=config.batch_size),
        "params": parameter_count(model),
        "seconds": elapsed,
        "config": asdict(config),
    }


def plot_training_curves(history: list[dict], output_path: Path, title: str) -> None:
    epochs = [item["epoch"] for item in history]
    train_mse = [item["train_mse"] for item in history]
    test_mse = [item["test_mse"] for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mse, label="Train MSE")
    plt.plot(epochs, test_mse, label="Test MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def evaluate_grid(model: nn.Module, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.5, 1.0, grid_size, dtype=np.float32)
    x1, x2 = np.meshgrid(xs, xs)
    coords = np.stack([x1.ravel(), x2.ravel()], axis=1)
    with torch.no_grad():
        pred = model(torch.tensor(coords, dtype=torch.float32)).cpu().numpy().reshape(grid_size, grid_size)
    target = np.sin(np.exp(x1) + x2)
    return x1, target, pred


def plot_fourier_spectra(model: nn.Module, output_path: Path, grid_size: int) -> None:
    _, target, pred = evaluate_grid(model, grid_size)
    target_fft = np.abs(np.fft.fftshift(np.fft.fft2(target)))
    pred_fft = np.abs(np.fft.fftshift(np.fft.fft2(pred)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(target_fft, cmap="magma")
    axes[0].set_title("Target Spectrum")
    axes[1].imshow(pred_fft, cmap="magma")
    axes[1].set_title("Model Spectrum")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_history_csv(history: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "name",
        "group",
        "n_qubits",
        "n_layers",
        "encoding",
        "train_mse",
        "test_mse",
        "params",
        "seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report_markdown(
    *,
    output_path: Path,
    seed: int,
    sweep_rows: list[dict],
    best_result: dict,
    final_result: dict | None,
) -> None:
    lines = [
        "# Problem 1 Sweep Summary",
        "",
        f"- Seed: `{seed}`",
        f"- Best sweep config: `{best_result['config']['name']}`",
        f"- Best sweep test MSE: `{best_result['test_mse']:.6f}`",
    ]
    if final_result is not None:
        lines.extend(
            [
                f"- Final run config: `{final_result['config']['name']}`",
                f"- Final run test MSE: `{final_result['test_mse']:.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Controlled Sweep",
            "",
            "| name | group | qubits | layers | encoding | train_mse | test_mse | params | seconds |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(sweep_rows, key=lambda item: (item["group"], item["test_mse"])):
        lines.append(
            f"| {row['name']} | {row['group']} | {row['n_qubits']} | {row['n_layers']} | "
            f"{row['encoding']} | {row['train_mse']:.6f} | {row['test_mse']:.6f} | {row['params']} | {row['seconds']:.2f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Problem 1 data reuploading sweep")
    parser.add_argument("--seed", type=int, required=True, help="Use the numeric part of your student ID")
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--preset", type=str, default="quick", choices=sorted(PRESET_CONFIG_NAMES))
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--grid-qubits", type=int, nargs="*", default=[2, 3, 4])
    parser.add_argument("--grid-layers", type=int, nargs="*", default=[2, 4, 6, 8])
    parser.add_argument("--grid-encodings", type=str, nargs="*", default=["rx_ry", "ry_rz"])
    parser.add_argument("--grid-lrs", type=float, nargs="*", default=[0.03])
    parser.add_argument("--device", type=str, default="default.qubit", choices=["default.qubit", "lightning.qubit"])
    parser.add_argument("--wandb", type=str, default="offline", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="QCAA HW1")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--configs", nargs="*", default=None, help="Subset of sweep configs to run")
    parser.add_argument("--run-final-best", action="store_true")
    parser.add_argument("--final-train-size", type=int, default=1000)
    parser.add_argument("--final-test-size", type=int, default=1000)
    parser.add_argument("--final-epochs", type=int, default=80)
    parser.add_argument("--final-eval-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    ensure_runtime_dirs()
    args = parse_args()
    set_seed(args.seed)
    train_x, train_y, test_x, test_y = build_dataset(args.n_train, args.n_test)

    results = []
    best_result = None
    if args.grid_search:
        sweep = build_grid_configs(
            qubits=args.grid_qubits,
            layers=args.grid_layers,
            encodings=args.grid_encodings,
            lrs=args.grid_lrs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
        )
        if not sweep:
            raise ValueError("Grid search produced no configs.")
    else:
        selected_names = args.configs if args.configs is not None else PRESET_CONFIG_NAMES[args.preset]
        sweep = [CONFIG_MAP[name] for name in selected_names if name in CONFIG_MAP]
        if not sweep:
            raise ValueError(f"No matching configs found in: {selected_names}")

    for base_cfg in sweep:
        config = RunConfig(
            name=base_cfg.name,
            n_qubits=base_cfg.n_qubits,
            n_layers=base_cfg.n_layers,
            encoding=base_cfg.encoding,
            group=base_cfg.group,
            lr=base_cfg.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_every=args.eval_every,
        )
        result = train_single_run(
            config=config,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            seed=args.seed,
            device_name=args.device,
            wandb_mode=args.wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )
        results.append(result)
        if best_result is None or result["test_mse"] < best_result["test_mse"]:
            best_result = result

        run_dir = OUTPUT_ROOT / config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        write_history_csv(result["history"], run_dir / "history.csv")
        plot_training_curves(result["history"], run_dir / "training_curves.png", f"Run: {config.name}")
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": result["config"],
                    "train_mse": result["train_mse"],
                    "test_mse": result["test_mse"],
                    "params": result["params"],
                    "seconds": result["seconds"],
                },
                handle,
                indent=2,
            )

    assert best_result is not None
    best_name = best_result["config"]["name"]
    best_dir = OUTPUT_ROOT / best_name
    plot_training_curves(best_result["history"], best_dir / "training_curves.png", f"Best Run: {best_name}")
    plot_fourier_spectra(best_result["model"], best_dir / "fourier_spectra.png", args.grid_size)

    summary_rows = []
    for result in results:
        cfg = result["config"]
        summary_rows.append(
            {
                "name": cfg["name"],
                "group": cfg["group"],
                "n_qubits": cfg["n_qubits"],
                "n_layers": cfg["n_layers"],
                "encoding": cfg["encoding"],
                "train_mse": result["train_mse"],
                "test_mse": result["test_mse"],
                "params": result["params"],
                "seconds": result["seconds"],
            }
        )
    write_summary_csv(summary_rows, OUTPUT_ROOT / "comparison.csv")

    final_result = None
    if args.run_final_best:
        set_seed(args.seed)
        final_train_x, final_train_y, final_test_x, final_test_y = build_dataset(args.final_train_size, args.final_test_size)
        best_cfg = best_result["config"]
        final_config = RunConfig(
            name=f"{best_cfg['name']}_final",
            n_qubits=best_cfg["n_qubits"],
            n_layers=best_cfg["n_layers"],
            encoding=best_cfg["encoding"],
            group="final",
            lr=best_cfg["lr"],
            epochs=args.final_epochs,
            batch_size=args.batch_size,
            eval_every=args.final_eval_every,
        )
        final_result = train_single_run(
            config=final_config,
            train_x=final_train_x,
            train_y=final_train_y,
            test_x=final_test_x,
            test_y=final_test_y,
            seed=args.seed,
            device_name=args.device,
            wandb_mode=args.wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
        )
        final_dir = OUTPUT_ROOT / final_config.name
        final_dir.mkdir(parents=True, exist_ok=True)
        write_history_csv(final_result["history"], final_dir / "history.csv")
        plot_training_curves(final_result["history"], final_dir / "training_curves.png", f"Final Run: {final_config.name}")
        plot_fourier_spectra(final_result["model"], final_dir / "fourier_spectra.png", args.grid_size)
        with (final_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": final_result["config"],
                    "train_mse": final_result["train_mse"],
                    "test_mse": final_result["test_mse"],
                    "params": final_result["params"],
                    "seconds": final_result["seconds"],
                },
                handle,
                indent=2,
            )
        maybe_log_wandb_summary(
            mode=args.wandb,
            entity=args.wandb_entity,
            project=args.wandb_project,
            seed=args.seed,
            best_name=final_config.name,
            best_test_mse=final_result["test_mse"],
            training_curve=final_dir / "training_curves.png",
            fourier_plot=final_dir / "fourier_spectra.png",
        )
    else:
        maybe_log_wandb_summary(
            mode=args.wandb,
            entity=args.wandb_entity,
            project=args.wandb_project,
            seed=args.seed,
            best_name=best_name,
            best_test_mse=best_result["test_mse"],
            training_curve=best_dir / "training_curves.png",
            fourier_plot=best_dir / "fourier_spectra.png",
        )

    write_report_markdown(
        output_path=OUTPUT_ROOT / "report_summary.md",
        seed=args.seed,
        sweep_rows=summary_rows,
        best_result=best_result,
        final_result=final_result,
    )

    print("Sweep finished.")
    print(f"Best config: {best_name}")
    print(f"Best train MSE: {best_result['train_mse']:.6f}")
    print(f"Best test MSE: {best_result['test_mse']:.6f}")
    if final_result is not None:
        print(f"Final run test MSE: {final_result['test_mse']:.6f}")
    print(f"Comparison table: {OUTPUT_ROOT / 'comparison.csv'}")


if __name__ == "__main__":
    main()
