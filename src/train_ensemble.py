"""
train_ensemble.py — Learn optimal blend weights for MLP, CNN, Attention CNN, and XGBoost
using the validation split to minimize Log Loss (Item 7).

Usage:
  python src/train_ensemble.py
"""

import sys
import pickle
import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss, classification_report

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.models.baseline_mlp  import BaselineMLP
from src.models.tactical_cnn  import TacticalCNN
from src.models.attention_cnn import AttentionCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_dataset(path: Path) -> TensorDataset:
    data = np.load(path)
    tensors = [
        torch.from_numpy(data["home_players"]).float(),
        torch.from_numpy(data["away_players"]).float(),
        torch.from_numpy(data["context"]).float(),
        torch.from_numpy(data["targets"]).long(),
    ]
    return TensorDataset(*tensors)


@torch.no_grad()
def get_nn_probs(model, loader, device) -> np.ndarray:
    model.eval()
    all_probs = []
    for home, away, ctx, _ in loader:
        if ctx.shape[1] > 100:
            ctx = ctx[:, :100]
        home, away, ctx = home.to(device), away.to(device), ctx.to(device)
        logits = model(home, away, ctx)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── 1. Load Validation Data ────────────────────────────────────────────────
    log.info("Loading validation split...")
    val_ds = load_dataset(config.VAL_NPZ)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    y_val = val_ds.tensors[3].numpy()

    # Context dim
    C_actual = val_ds.tensors[2].shape[1]
    C_nn = 100 if C_actual > 100 else C_actual

    # ── 2. Load Checkpoints & Predict on Val ──────────────────────────────────
    log.info("Generating predictions on validation set...")
    
    # MLP
    mlp = BaselineMLP(F=config.F, C=C_nn).to(device)
    mlp_path = config.OUTPUTS_MODELS / "baseline_mlp_best.pt"
    if mlp_path.exists():
        mlp.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
        mlp_probs = get_nn_probs(mlp, val_loader, device)
    else:
        mlp_probs = None

    # CNN
    cnn = TacticalCNN(F=config.F, C=C_nn).to(device)
    cnn_path = config.OUTPUTS_MODELS / "tactical_cnn_best.pt"
    if cnn_path.exists():
        cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        cnn_probs = get_nn_probs(cnn, val_loader, device)
    else:
        cnn_probs = None

    # Attention CNN
    att = AttentionCNN(F=config.F, C=C_nn).to(device)
    att_path = config.OUTPUTS_MODELS / "attention_cnn_best.pt"
    if att_path.exists():
        att.load_state_dict(torch.load(att_path, map_location=device, weights_only=True))
        att_probs = get_nn_probs(att, val_loader, device)
    else:
        att_probs = None

    # XGBoost
    xgb_path = config.OUTPUTS_MODELS / "xgboost_best.pkl"
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        X_val = val_ds.tensors[2].numpy()
        xgb_probs = xgb_model.predict_proba(X_val)
    else:
        xgb_probs = None

    # ── 3. Align available models ──────────────────────────────────────────────
    models_available = []
    probs_list = []
    
    if mlp_probs is not None:
        models_available.append("baseline_mlp")
        probs_list.append(mlp_probs)
    if cnn_probs is not None:
        models_available.append("tactical_cnn")
        probs_list.append(cnn_probs)
    if att_probs is not None:
        models_available.append("attention_cnn")
        probs_list.append(att_probs)
    if xgb_probs is not None:
        models_available.append("xgboost")
        probs_list.append(xgb_probs)

    if not models_available:
        log.error("No trained models found to ensemble!")
        return

    log.info(f"Models participating in the ensemble: {models_available}")

    # Stack probabilities to shape (num_samples, num_models, 3)
    # where probs_list has shape (num_models, num_samples, 3)
    probs_stack = np.stack(probs_list, axis=1)

    # ── 4. Find Optimal Blending Weights ───────────────────────────────────────
    # Objective function: minimize Log Loss
    def loss_func(weights):
        # Normalize weights to sum to 1
        w = weights / np.sum(weights)
        # Weighted combination of probabilities
        blend_probs = np.tensordot(probs_stack, w, axes=(1, 0))
        return log_loss(y_val, blend_probs)

    # Constraints: weights sum to 1, all weights >= 0
    cons = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
    bounds = [(0.0, 1.0)] * len(models_available)
    init_weights = np.ones(len(models_available)) / len(models_available)

    log.info("Optimizing ensemble weights...")
    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
    
    opt_weights = res.x / np.sum(res.x)
    weight_map = {name: float(w) for name, w in zip(models_available, opt_weights)}
    
    log.info("\n=== Learned Blend Weights ===")
    for name, w in weight_map.items():
        log.info(f"  {name:<15}: {w:.4f}")
    
    # Save the learned weights
    weights_path = config.OUTPUTS_MODELS / "ensemble_weights.pkl"
    with open(weights_path, "wb") as f:
        pickle.dump(weight_map, f)
    log.info(f"\nSaved optimal weights → {weights_path}")

    # Evaluate blended model on validation set
    val_blend_probs = np.tensordot(probs_stack, opt_weights, axes=(1, 0))
    val_blend_preds = val_blend_probs.argmax(axis=1)
    val_blend_loss = log_loss(y_val, val_blend_probs)
    val_blend_acc = accuracy_score(y_val, val_blend_preds)
    log.info(f"Validation Blended Loss: {val_blend_loss:.4f} | Accuracy: {val_blend_acc:.4f}")

    # ── 5. Evaluate Stacking on Test Split if Available ───────────────────────
    if config.TEST_NPZ.exists():
        log.info("\nEvaluating ensemble on test split...")
        test_ds = load_dataset(config.TEST_NPZ)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        y_test = test_ds.tensors[3].numpy()
        X_test = test_ds.tensors[2].numpy()

        test_probs_list = []
        if "baseline_mlp" in weight_map:
            test_probs_list.append(get_nn_probs(mlp, test_loader, device))
        if "tactical_cnn" in weight_map:
            test_probs_list.append(get_nn_probs(cnn, test_loader, device))
        if "attention_cnn" in weight_map:
            test_probs_list.append(get_nn_probs(att, test_loader, device))
        if "xgboost" in weight_map:
            test_probs_list.append(xgb_model.predict_proba(X_test))

        test_probs_stack = np.stack(test_probs_list, axis=1)
        test_blend_probs = np.tensordot(test_probs_stack, opt_weights, axes=(1, 0))
        test_blend_preds = test_blend_probs.argmax(axis=1)
        
        test_blend_loss = log_loss(y_test, test_blend_probs)
        test_blend_acc = accuracy_score(y_test, test_blend_preds)
        
        log.info("\n=== Blended Test Results ===")
        log.info(f"Blended Accuracy: {test_blend_acc:.4f}")
        log.info(f"Blended Log Loss: {test_blend_loss:.4f}")
        log.info("\n" + classification_report(y_test, test_blend_preds, target_names=config.RESULT_NAMES))


if __name__ == "__main__":
    main()
