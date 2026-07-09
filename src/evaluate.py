"""
evaluate.py — Evaluate all trained models and generate visualisation plots.

Produces:
  outputs/plots/cm_{model}.png              — Confusion matrix
  outputs/plots/loss_{model}.png            — Train/val loss curves
  outputs/plots/model_comparison.png        — Side-by-side metrics bar chart
  outputs/plots/attention_weights.png       — Model 3: avg attention per position
  outputs/plots/class_distribution.png      — Target class distribution
  outputs/predictions/val_predictions.csv   — Per-match prediction details
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix, accuracy_score, log_loss,
    classification_report, f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.models.baseline_mlp  import BaselineMLP
from src.models.tactical_cnn  import TacticalCNN
from src.models.attention_cnn import AttentionCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = config.RESULT_NAMES  # ["Home Win", "Draw", "Away Win"]
PALETTE = {"baseline_mlp": "#4A90D9", "tactical_cnn": "#E67E22", "attention_cnn": "#2ECC71", "xgboost": "#9B59B6"}



def load_eval_dataset() -> tuple[TensorDataset, str]:
    """Load evaluation dataset. Prefers TEST_NPZ if available, otherwise falls back to VAL_NPZ."""
    if config.TEST_NPZ.exists():
        log.info(f"Loading test split for evaluation: {config.TEST_NPZ}")
        data = np.load(config.TEST_NPZ)
        split_name = "test"
    else:
        log.info(f"Loading val split for evaluation: {config.VAL_NPZ}")
        data = np.load(config.VAL_NPZ)
        split_name = "val"

    ds = TensorDataset(
        torch.from_numpy(data["home_players"]).float(),
        torch.from_numpy(data["away_players"]).float(),
        torch.from_numpy(data["context"]).float(),
        torch.from_numpy(data["targets"]).long(),
    )
    return ds, split_name



def load_model(model_class, name: str, C: int) -> torch.nn.Module | None:
    path = config.OUTPUTS_MODELS / f"{name}_best.pt"
    if not path.exists():
        log.warning(f"  Model checkpoint not found: {path}")
        return None
    model = model_class(F=config.F, C=C)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    log.info(f"  Loaded {name} from {path}")
    return model


@torch.no_grad()
def get_predictions(model: torch.nn.Module,
                    loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred, y_prob)."""
    all_true, all_pred, all_prob = [], [], []

    for home, away, ctx, targets in loader:
        if ctx.shape[1] > 100:
            ctx = ctx[:, :100]
        logits = model(home, away, ctx)
        probs  = torch.softmax(logits, dim=1).numpy()
        preds  = probs.argmax(axis=1)

        all_true.extend(targets.numpy())
        all_pred.extend(preds)
        all_prob.extend(probs)

    return np.array(all_true), np.array(all_pred), np.array(all_prob)


@torch.no_grad()
def get_attention_weights(model: AttentionCNN,
                          loader: DataLoader) -> np.ndarray:
    """Extract average attention weights across all validation matches."""
    all_weights = []
    for home, away, ctx, _ in loader:
        if ctx.shape[1] > 100:
            ctx = ctx[:, :100]
        _ = model(home.to(next(model.parameters()).device),
                  away.to(next(model.parameters()).device),
                  ctx.to(next(model.parameters()).device))
        w = model.get_attention_weights()
        # Average home and away weights
        avg_w = (w["home"] + w["away"]) / 2
        all_weights.append(avg_w.cpu().numpy())
    return np.concatenate(all_weights, axis=0).mean(axis=0)  # shape: (11,)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[f"Pred {c}" for c in CLASS_NAMES],
                yticklabels=[f"Actual {c}" for c in CLASS_NAMES])
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    out = config.OUTPUTS_PLOTS / f"cm_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_loss_curves(history: dict, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#4A90D9", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#E74C3C", linewidth=2)
    if "best_epoch" in history:
        ax.axvline(history["best_epoch"], color="green", linestyle="--", alpha=0.7,
                   label=f'Best (epoch {history["best_epoch"]})')
    ax.set_title(f"Loss Curves — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CrossEntropy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = config.OUTPUTS_PLOTS / f"loss_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_model_comparison(results: dict) -> None:
    """Side-by-side bar chart comparing all 3 models."""
    model_names = list(results.keys())
    metrics     = ["Accuracy", "Log Loss", "F1-Draw", "Macro F1"]
    n_models    = len(model_names)
    n_metrics   = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        vals   = [results[m].get(metric, 0) for m in model_names]
        colors = [PALETTE.get(m, "#999") for m in model_names]
        bars   = ax.bar(range(n_models), vals, color=colors, edgecolor="white", width=0.6)
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([m.replace("_", "\n") for m in model_names], fontsize=9)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison — Validation Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = config.OUTPUTS_PLOTS / "model_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_attention_weights(weights: np.ndarray) -> None:
    """Bar chart of avg attention weight per canonical position slot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pos_labels = config.POSITION_ORDER
    colors = plt.cm.RdYlGn(weights / weights.max())
    bars = ax.bar(range(len(pos_labels)), weights, color=colors, edgecolor="white")
    ax.set_xticks(range(len(pos_labels)))
    ax.set_xticklabels(pos_labels, fontsize=11)
    ax.set_title("Attention CNN — Average Attention Weight per Position Slot",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Average α weight")
    ax.set_xlabel("Canonical Position Slot (GK → ST)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = config.OUTPUTS_PLOTS / "attention_weights.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_class_distribution(y_train: np.ndarray, y_val: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (split_name, targets) in zip(axes, [("Train", y_train), ("Val", y_val)]):
        counts = np.bincount(targets, minlength=3)
        colors = ["#3498DB", "#F39C12", "#E74C3C"]
        ax.bar(CLASS_NAMES, counts, color=colors, edgecolor="white")
        ax.set_title(f"{split_name} Set — Class Distribution", fontweight="bold")
        ax.set_ylabel("Count")
        for i, c in enumerate(counts):
            ax.text(i, c + 5, str(c), ha="center", fontsize=11)
    plt.tight_layout()
    out = config.OUTPUTS_PLOTS / "class_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config.OUTPUTS_PLOTS.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_PREDICTIONS.mkdir(parents=True, exist_ok=True)

    # Load eval dataset (Item 4)
    eval_ds, split_name = load_eval_dataset()
    loader = DataLoader(eval_ds, batch_size=64, shuffle=False)


    # Actual context dim from loaded data
    C_actual = eval_ds[0][2].shape[0]


    # Load training history
    hist_path = config.OUTPUTS_MODELS / "training_histories.pkl"
    histories = {}
    if hist_path.exists():
        with open(hist_path, "rb") as f:
            histories = pickle.load(f)

    # Load training targets for class distribution
    train_data = np.load(config.TRAIN_NPZ)
    y_train    = train_data["targets"]
    
    if split_name == "test":
        eval_data = np.load(config.TEST_NPZ)
    else:
        eval_data = np.load(config.VAL_NPZ)
    y_eval_all = eval_data["targets"]

    # Class distribution plot
    plot_class_distribution(y_train, y_eval_all)


    model_registry = {
        "baseline_mlp":  (BaselineMLP,    {}),
        "tactical_cnn":  (TacticalCNN,    {}),
        "attention_cnn": (AttentionCNN,   {}),
        "xgboost":       (None,           {}),
    }


    all_results = {}
    all_preds   = {}

    print("\n" + "="*70)
    print(f"{'Model':<20} {'Accuracy':>10} {'Log Loss':>10} "
          f"{'F1-Home':>9} {'F1-Draw':>9} {'F1-Away':>9} {'Macro F1':>10}")
    print("="*70)

    for model_name, (model_class, kwargs) in model_registry.items():
        if model_name == "xgboost":
            path = config.OUTPUTS_MODELS / "xgboost_best.pkl"
            if not path.exists():
                log.warning(f"  Model checkpoint not found: {path}")
                continue
            with open(path, "rb") as f:
                model = pickle.load(f)
            log.info(f"  Loaded xgboost from {path}")
            
            # Direct prediction on evaluation features
            X_eval = eval_ds.tensors[2].numpy()
            y_true = eval_ds.tensors[3].numpy()
            y_prob = model.predict_proba(X_eval)
            y_pred = model.predict(X_eval)
        else:
            C_nn = 100 if C_actual > 100 else C_actual
            model = load_model(model_class, model_name, C=C_nn)
            if model is None:
                continue
            y_true, y_pred, y_prob = get_predictions(model, loader)


        # Metrics
        acc      = accuracy_score(y_true, y_pred)
        ll       = log_loss(y_true, y_prob)
        report   = classification_report(
            y_true, y_pred, target_names=CLASS_NAMES,
            zero_division=0, output_dict=True
        )
        f1_home  = report["Home Win"]["f1-score"]
        f1_draw  = report["Draw"]["f1-score"]
        f1_away  = report["Away Win"]["f1-score"]
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        metrics = {
            "Accuracy": acc,
            "Log Loss": ll,
            "F1-Home":  f1_home,
            "F1-Draw":  f1_draw,
            "F1-Away":  f1_away,
            "Macro F1": macro_f1,
        }
        all_results[model_name] = metrics
        all_preds[model_name]   = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

        print(f"{model_name:<20} {acc:>10.4f} {ll:>10.4f} "
              f"{f1_home:>9.4f} {f1_draw:>9.4f} {f1_away:>9.4f} {macro_f1:>10.4f}")

        # Plots
        plot_confusion_matrix(y_true, y_pred, model_name)
        if model_name in histories:
            plot_loss_curves(histories[model_name], model_name)

        # Attention weights (Model 3 only)
        if model_name == "attention_cnn":
            avg_weights = get_attention_weights(model, loader)
            plot_attention_weights(avg_weights)

    print("="*70)

    if all_results:
        plot_model_comparison(all_results)

    # Save per-match predictions CSV
    if all_preds:
        first_model = next(iter(all_preds))
        y_true = all_preds[first_model]["y_true"]
        rows = []
        for i in range(len(y_true)):
            row = {"true_result": config.RESULT_NAMES[y_true[i]]}
            for mname, pdata in all_preds.items():
                row[f"{mname}_pred"]   = config.RESULT_NAMES[pdata["y_pred"][i]]
                row[f"{mname}_home%"]  = round(pdata["y_prob"][i][0] * 100, 1)
                row[f"{mname}_draw%"]  = round(pdata["y_prob"][i][1] * 100, 1)
                row[f"{mname}_away%"]  = round(pdata["y_prob"][i][2] * 100, 1)
            rows.append(row)

        pred_csv = config.OUTPUTS_PREDICTIONS / f"{split_name}_predictions.csv"
        pd.DataFrame(rows).to_csv(pred_csv, index=False)
        log.info(f"\nSaved {split_name} predictions → {pred_csv}")


    log.info("\n✓ evaluate.py complete.")


if __name__ == "__main__":
    main()
