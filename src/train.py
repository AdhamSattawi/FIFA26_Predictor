"""
train.py — Train all 3 models on the processed feature tensors.

Usage:
  python src/train.py                     # train all models
  python src/train.py --model mlp         # train only the baseline MLP
  python src/train.py --model cnn         # train only the tactical CNN
  python src/train.py --model attention   # train only the attention CNN
  python src/train.py --model all         # train all (default)
"""

import sys
import argparse
import logging
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.models.baseline_mlp  import BaselineMLP
from src.models.tactical_cnn  import TacticalCNN
from src.models.attention_cnn import AttentionCNN, compute_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


# ── Dataset helper ────────────────────────────────────────────────────────────

def load_split(path: Path) -> TensorDataset:
    """Load an .npz split into a TensorDataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/processing/feature_engineering.py first."
        )
    data = np.load(path)
    return TensorDataset(
        torch.from_numpy(data["home_players"]).float(),
        torch.from_numpy(data["away_players"]).float(),
        torch.from_numpy(data["context"]).float(),
        torch.from_numpy(data["targets"]).long(),
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    scaler=None) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for home, away, ctx, targets in loader:
        home, away, ctx, targets = (
            home.to(device), away.to(device), ctx.to(device), targets.to(device)
        )
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(home, away, ctx)
                loss   = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(home, away, ctx)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for home, away, ctx, targets in loader:
        home, away, ctx, targets = (
            home.to(device), away.to(device), ctx.to(device), targets.to(device)
        )
        logits = model(home, away, ctx)
        loss   = criterion(logits, targets)
        total_loss += loss.item() * len(targets)
        preds   = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total   += len(targets)

    return total_loss / total, correct / total


def train_model(model: nn.Module,
                model_name: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                patience: int,
                max_epochs: int,
                use_lr_scheduler: bool = False,
                class_weights: torch.Tensor | None = None) -> dict:
    """
    Full training loop with early stopping and checkpointing.
    Returns training history dict.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Training: {model_name}")
    log.info(f"{'='*60}")

    model = model.to(device)

    # Loss
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        log.info(f"  Class weights: {class_weights.numpy().round(3)}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    wd = config.WEIGHT_DECAY if use_lr_scheduler else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=wd)

    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )

    # Mixed precision (GPU only)
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0
    save_path     = config.OUTPUTS_MODELS / f"{model_name}_best.pt"
    config.OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, scaler=amp_scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler:
            scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch    = epoch
            no_improve    = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        marker = " ◄ BEST" if improved else ""
        elapsed = time.time() - t0
        log.info(
            f"  Epoch {epoch:3d}/{max_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if no_improve >= patience:
            log.info(f"  Early stopping at epoch {epoch} (best: epoch {best_epoch}).")
            break

    log.info(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    log.info(f"  Saved best model → {save_path}")

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = best_val_loss

    return model, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train FIFA26 prediction models")
    parser.add_argument("--model", default="all",
                        choices=["mlp", "cnn", "attention", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading feature tensors …")
    train_ds = load_split(config.TRAIN_NPZ)
    val_ds   = load_split(config.VAL_NPZ)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    log.info(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # Load actual context dimension from saved features
    sample_ctx = train_ds[0][2]
    C_actual   = sample_ctx.shape[0]
    log.info(f"Context dim C = {C_actual}")

    # Compute class weights for Model 3 from training targets
    all_targets = train_ds.tensors[3]
    class_weights = compute_class_weights(all_targets)
    log.info(f"Class distribution — H:{(all_targets==0).sum()} "
             f"D:{(all_targets==1).sum()} A:{(all_targets==2).sum()}")

    all_histories = {}

    # ── Model 1: Baseline MLP ─────────────────────────────────────────────────
    if args.model in ("mlp", "all"):
        mlp = BaselineMLP(F=config.F, C=C_actual)
        mlp, hist = train_model(
            mlp, "baseline_mlp",
            train_loader, val_loader, device,
            patience=config.EARLY_STOP_PATIENCE,
            max_epochs=100,
        )
        all_histories["baseline_mlp"] = hist

    # ── Model 2: Tactical CNN ─────────────────────────────────────────────────
    if args.model in ("cnn", "all"):
        cnn = TacticalCNN(F=config.F, C=C_actual)
        cnn, hist = train_model(
            cnn, "tactical_cnn",
            train_loader, val_loader, device,
            patience=config.CNN_PATIENCE,
            max_epochs=150,
        )
        all_histories["tactical_cnn"] = hist

    # ── Model 3: Attention CNN ────────────────────────────────────────────────
    if args.model in ("attention", "all"):
        att = AttentionCNN(F=config.F, C=C_actual)
        att, hist = train_model(
            att, "attention_cnn",
            train_loader, val_loader, device,
            patience=config.ATT_PATIENCE,
            max_epochs=200,
            use_lr_scheduler=True,
            class_weights=class_weights,
        )
        all_histories["attention_cnn"] = hist

    # ── Save histories ────────────────────────────────────────────────────────
    hist_path = config.OUTPUTS_MODELS / "training_histories.pkl"
    with open(hist_path, "wb") as f:
        pickle.dump(all_histories, f)
    log.info(f"\n✓ Training histories saved → {hist_path}")
    log.info("✓ Training complete. Run src/evaluate.py for results.")


if __name__ == "__main__":
    main()
