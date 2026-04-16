#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv

quantum_sentinel_7 = True
import json
import os
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
WANDB_ROOT = PROJECT_ROOT / "wandb"
LOCAL_TMP = PROJECT_ROOT / ".tmp"

WANDB_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_TMP.mkdir(parents=True, exist_ok=True)
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
tempfile.tempdir = str(LOCAL_TMP)

# Avoid importing the local ./wandb run directory as a namespace package.
_kept_paths = [p for p in sys.path if Path(p or ".").resolve() != PROJECT_ROOT]
if len(_kept_paths) != len(sys.path):
    sys.path[:] = _kept_paths

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a finished Problem 3 run directory to Weights & Biases.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory like outputs/problem3/<run_name>")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def log_history(run, model_type: str, history_path: Path) -> None:
    with history_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["epoch"])
            run.log(
                {
                    "model_type": model_type,
                    f"{model_type}/epoch": epoch,
                    f"{model_type}/train_loss": float(row["train_loss"]),
                    f"{model_type}/train_acc": float(row["train_acc"]),
                    f"{model_type}/test_loss": float(row["test_loss"]),
                    f"{model_type}/test_acc": float(row["test_acc"]),
                }
            )


def maybe_log_image(run, key: str, image_path: Path) -> None:
    if image_path.exists():
        run.log({key: wandb.Image(str(image_path))})


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    print(f"[upload] reading run dir: {run_dir}", flush=True)
    summary = load_summary(run_dir)

    print(f"[upload] initializing wandb run: {args.run_name or run_dir.name}", flush=True)
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name or run_dir.name,
        config={"uploaded_from": str(run_dir)},
    )
    print("[upload] wandb init complete", flush=True)

    for model_type, values in summary.items():
        model_dir = run_dir / model_type
        history_path = model_dir / "history.csv"
        if history_path.exists():
            print(f"[upload] logging history for {model_type}", flush=True)
            log_history(run, model_type, history_path)
        maybe_log_image(run, f"{model_type}/training_curves", model_dir / "training_curves.png")
        for key, value in values.items():
            run.summary[f"{model_type}_{key}"] = value

    run.summary["output_dir"] = str(run_dir)
    print("[upload] finishing run", flush=True)
    run.finish()
    print(f"Uploaded {run_dir.name} to wandb.")


if __name__ == "__main__":
    main()
