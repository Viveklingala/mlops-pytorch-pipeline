"""Training entrypoint for the CIFAR-10 classifier.

Reads all hyperparameters from a YAML config (mounted as a ConfigMap in
Kubernetes, or passed via --config locally), logs structured JSON lines to
stdout for every epoch, checkpoints on validation-loss improvement, and
supports early stopping.

Usage:
    python src/train.py --config configs/training_config.yaml
    python src/train.py  # falls back to /app/configs/training_config.yaml,
                          # then configs/training_config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_config_path(cli_arg: str | None) -> Path:
    """Precedence: --config flag > CONFIG_PATH env var > known defaults."""
    candidates = []
    if cli_arg:
        candidates.append(Path(cli_arg))
    if os.environ.get("CONFIG_PATH"):
        candidates.append(Path(os.environ["CONFIG_PATH"]))
    candidates += [Path("/app/configs/training_config.yaml"), Path("configs/training_config.yaml")]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No training config found. Tried: {[str(c) for c in candidates]}"
    )


def log(event: dict) -> None:
    """Structured JSON-lines logging to stdout (easy to grep / ship to a log
    aggregator from a Kubernetes Job's stdout)."""
    print(json.dumps(event), flush=True)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 classifier")
    parser.add_argument("--config", type=str, default=None, help="Path to training_config.yaml")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on batches/epoch, useful for smoke-testing the "
        "training image without a full epoch (also settable via MAX_BATCHES env var).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(str(config_path))
    log({"event": "config_loaded", "path": str(config_path)})

    max_batches = args.max_batches or (
        int(os.environ["MAX_BATCHES"]) if os.environ.get("MAX_BATCHES") else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log({"event": "device_selected", "device": str(device)})

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
    )

    if max_batches:
        # Truncate loaders for CI / smoke tests so `docker run` verification
        # doesn't have to wait through a full CIFAR-10 epoch.
        from itertools import islice

        train_loader = list(islice(train_loader, max_batches))
        val_loader = list(islice(val_loader, max(1, max_batches // 4)))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", config["output"]["checkpoint_dir"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log(
            {
                "event": "epoch_complete",
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "duration_sec": round(time.time() - start, 2),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "architecture": config["model"]["architecture"],
                    "num_classes": config["model"]["num_classes"],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            log({"event": "checkpoint_saved", "path": str(save_path)})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log({"event": "early_stopping", "epoch": epoch + 1})
                break

    log({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)})


if __name__ == "__main__":
    main()
