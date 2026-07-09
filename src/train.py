"""
train.py — Train all 3 models on the processed feature tensors.

Usage:
  python src/train.py                        # train all models (focal loss)
  python src/train.py --model mlp            # train only the baseline MLP
  python src/train.py --model cnn            # train only the tactical CNN
  python src/train.py --model attention      # train only the attention CNN
  python src/train.py --model all            # train all (default)
  python src/train.py --loss cross_entropy   # use plain CrossEntropyLoss instead
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


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples so the model focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2.0  : standard focal loss exponent
    class_weights: same as CrossEntropyLoss weight parameter

    This combats class imbalance better than class weighting alone because
    it also suppresses confident correct predictions (e.g. obvious Home Wins)
    and forces attention on ambiguous matches (draws, upsets).
    """
    def __init__(self, gamma: float = 2.0,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma   = gamma
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None
            else torch.ones(3)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                 weights: torch.Tensor | None = None) -> torch.Tensor:
        # Standard cross-entropy with class weights gives log(p_t)
        ce = nn.functional.cross_entropy(
            logits, targets,
            weight=self.class_weights.to(logits.device),
            reduction="none"
        )
        # p_t = exp(-ce) — probability assigned to the correct class
        pt = torch.exp(-ce)
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce
        if weights is not None:
            loss = loss * weights
        return loss.mean()



# ── Dataset helper ────────────────────────────────────────────────────────────

def load_split(path: Path) -> TensorDataset:
    """Load an .npz split into a TensorDataset (Item 12: handles tournament weights)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/processing/feature_engineering.py first."
        )
    data = np.load(path)
    context_np = data["context"]
    # Slice context to original 100 features for deep learning models
    if context_np.shape[1] > 100:
        context_np = context_np[:, :100]
    tensors = [
        torch.from_numpy(data["home_players"]).float(),
        torch.from_numpy(data["away_players"]).float(),
        torch.from_numpy(context_np).float(),
        torch.from_numpy(data["targets"]).long(),
    ]
    if "weights" in data:
        tensors.append(torch.from_numpy(data["weights"]).float())
    return TensorDataset(*tensors)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor,
                 weights: torch.Tensor | None, criterion: nn.Module) -> torch.Tensor:
    """Helper to compute sample-weighted loss (Item 12)."""
    if isinstance(criterion, FocalLoss):
        return criterion(logits, targets, weights)
    
    # Fallback for CrossEntropyLoss
    ce = nn.functional.cross_entropy(
        logits, targets,
        weight=criterion.weight,
        reduction="none"
    )
    if weights is not None:
        ce = ce * weights
    return ce.mean()



# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    scaler=None,
                    mixup_alpha: float = 0.0,
                    noise_std: float = 0.0) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        home, away, ctx, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        weights = batch[4].to(device) if len(batch) > 4 else None

        # Add Gaussian noise to context features during training (Item 8)
        if noise_std > 0.0:
            noise = torch.randn_like(ctx) * noise_std
            ctx = ctx + noise

        # Apply Mixup data augmentation during training (Item 8)
        if mixup_alpha > 0.0 and np.random.rand() < 0.5:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = home.size(0)
            index = torch.randperm(batch_size).to(device)
            
            home_mixed = lam * home + (1.0 - lam) * home[index]
            away_mixed = lam * away + (1.0 - lam) * away[index]
            ctx_mixed  = lam * ctx + (1.0 - lam) * ctx[index]
            
            targets_a, targets_b = targets, targets[index]
            weights_b = weights[index] if weights is not None else None
            
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(home_mixed, away_mixed, ctx_mixed)
                    loss = lam * compute_loss(logits, targets_a, weights, criterion) + \
                           (1.0 - lam) * compute_loss(logits, targets_b, weights_b, criterion)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(home_mixed, away_mixed, ctx_mixed)
                loss = lam * compute_loss(logits, targets_a, weights, criterion) + \
                       (1.0 - lam) * compute_loss(logits, targets_b, weights_b, criterion)
                loss.backward()
                optimizer.step()
        else:
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(home, away, ctx)
                    loss   = compute_loss(logits, targets, weights, criterion)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(home, away, ctx)
                loss   = compute_loss(logits, targets, weights, criterion)
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

    for batch in loader:
        home, away, ctx, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        weights = batch[4].to(device) if len(batch) > 4 else None
        
        logits = model(home, away, ctx)
        loss   = compute_loss(logits, targets, weights, criterion)
        
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
                class_weights: torch.Tensor | None = None,
                loss_type: str = "focal",
                mixup_alpha: float = 0.0,
                noise_std: float = 0.0) -> tuple:
    """
    Full training loop with early stopping and checkpointing.
    Returns (model, history) tuple.
    """

    log.info(f"\n{'='*60}")
    log.info(f"Training: {model_name}")
    log.info(f"{'='*60}")

    model = model.to(device)

    # Loss — class weights applied to all models (Item 2)
    if class_weights is not None:
        log.info(f"  Class weights: {class_weights.numpy().round(3)}")

    if loss_type == "focal":
        criterion = FocalLoss(gamma=2.0, class_weights=class_weights)
        log.info("  Loss: FocalLoss(gamma=2.0) with class weights")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )
        log.info("  Loss: CrossEntropyLoss with class weights")

    # Optimizer — weight_decay applied to ALL models unconditionally (Item 3)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY  # was 0.0 for MLP/CNN — now always 1e-4
    )

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
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=amp_scaler, mixup_alpha=mixup_alpha, noise_std=noise_std
        )
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
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    history["best_epoch"]    = best_epoch
    history["best_val_loss"] = best_val_loss

    return model, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train FIFA26 prediction models")
    parser.add_argument("--model", default="all",
                        choices=["mlp", "cnn", "attention", "all"])
    parser.add_argument("--loss", default="focal",
                        choices=["focal", "cross_entropy"],
                        help="Loss function: focal (default) or cross_entropy")
    parser.add_argument("--mixup-alpha", default=0.2, type=float,
                        help="Beta distribution alpha parameter for Mixup data augmentation (Item 8)")
    parser.add_argument("--noise-std", default=0.05, type=float,
                        help="Standard deviation of Gaussian noise added to context features (Item 8)")
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

    # Compute class weights from training targets — applied to ALL models (Item 2)
    all_targets = train_ds.tensors[3]
    class_weights = compute_class_weights(all_targets)
    log.info(f"Class weights: {class_weights.numpy().round(3)}")
    log.info(f"Class distribution — H:{(all_targets==0).sum()} "
             f"D:{(all_targets==1).sum()} A:{(all_targets==2).sum()}")
    log.info(f"Loss function: {args.loss}")

    all_histories = {}

    # ── Model 1: Baseline MLP ─────────────────────────────────────────────────
    if args.model in ("mlp", "all"):
        mlp = BaselineMLP(F=config.F, C=C_actual)
        mlp, hist = train_model(
            mlp, "baseline_mlp",
            train_loader, val_loader, device,
            patience=config.EARLY_STOP_PATIENCE,
            max_epochs=100,
            use_lr_scheduler=True,
            class_weights=class_weights,   # Item 2: was None
            loss_type=args.loss,
            mixup_alpha=args.mixup_alpha,
            noise_std=args.noise_std,
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
            use_lr_scheduler=True,
            class_weights=class_weights,   # Item 2: was None
            loss_type=args.loss,
            mixup_alpha=args.mixup_alpha,
            noise_std=args.noise_std,
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
            loss_type=args.loss,
            mixup_alpha=args.mixup_alpha,
            noise_std=args.noise_std,
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
