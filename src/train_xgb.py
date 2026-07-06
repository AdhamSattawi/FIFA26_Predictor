"""
train_xgb.py — Train an XGBoost model on context features as a tabular baseline (Item 6).

Usage:
  python src/train_xgb.py
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, log_loss

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_context_data(npz_path: Path):
    """Load context features and targets from an NPZ file."""
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found. Run feature_engineering.py first.")
    data = np.load(npz_path)
    return data["context"], data["targets"]


def main():
    import xgboost as xgb

    log.info("Loading context datasets...")
    X_train, y_train = load_context_data(config.TRAIN_NPZ)
    X_val, y_val = load_context_data(config.VAL_NPZ)
    
    # Try to load test dataset if it exists (Item 4)
    has_test = config.TEST_NPZ.exists()
    if has_test:
        X_test, y_test = load_context_data(config.TEST_NPZ)
        log.info(f"Loaded train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    else:
        log.info(f"Loaded train={X_train.shape}, val={X_val.shape}")

    # Calculate class weights for sample_weight (Item 2)
    # w_c = N_total / (3 * N_c)
    classes, counts = np.unique(y_train, return_counts=True)
    n_total = len(y_train)
    class_weight_map = {c: n_total / (3 * count) for c, count in zip(classes, counts)}
    sample_weights = np.array([class_weight_map[y] for y in y_train])
    
    log.info(f"Class distribution: H={counts[0]} D={counts[1]} A={counts[2]}")
    log.info(f"Class weights applied to training: {list(class_weight_map.values())}")

    # Define XGBoost model (hyperparameters optimized for international football dataset)
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.03,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=config.SEED,
        eval_metric="mlogloss",
        early_stopping_rounds=30
    )

    log.info("Training XGBoost Classifier...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluate on Validation set
    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_loss = log_loss(y_val, val_probs)
    
    log.info("\n=== Validation Results ===")
    log.info(f"Accuracy: {val_acc:.4f}")
    log.info(f"Log Loss: {val_loss:.4f}")
    log.info("\n" + classification_report(y_val, val_preds, target_names=config.RESULT_NAMES))

    # Evaluate on Test set if available
    if has_test:
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        test_loss = log_loss(y_test, test_probs)
        
        log.info("\n=== Test Results ===")
        log.info(f"Accuracy: {test_acc:.4f}")
        log.info(f"Log Loss: {test_loss:.4f}")
        log.info("\n" + classification_report(y_test, test_preds, target_names=config.RESULT_NAMES))

    # Save the model
    model_dir = config.OUTPUTS_MODELS
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON for portability and compatibility with xgboost
    model_path_json = model_dir / "xgboost_best.json"
    model.save_model(model_path_json)
    log.info(f"Saved model to {model_path_json}")
    
    # Save as PKL for compatibility with sklearn wrapper loading
    model_path_pkl = model_dir / "xgboost_best.pkl"
    with open(model_path_pkl, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Saved sklearn wrapper to {model_path_pkl}")


if __name__ == "__main__":
    main()
