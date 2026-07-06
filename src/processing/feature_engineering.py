"""
feature_engineering.py — Build and normalise the final feature tensors
for all 3 model architectures.

Reads:
  data/processed/full_dataset.csv         — Gulati context + match metadata
  data/processed/player_matrices.pkl      — raw (11, F) player matrices per match

Writes:
  data/features/train_features.npz        — training set
  data/features/val_features.npz          — validation set
  outputs/models/scalers.pkl              — fitted StandardScalers
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_dataset():
    """Load the full processed dataset and player matrices."""
    log.info(f"Loading full dataset from {config.FULL_DATASET_CSV} …")
    df = pd.read_csv(config.FULL_DATASET_CSV, parse_dates=["date"])

    # Item 9: Compute is_knockout
    def check_knockout(row):
        if row["is_world_cup"] != 1:
            return 0
        dt = row["date"]
        y = dt.year
        if y == 2014 and dt >= pd.Timestamp("2014-06-28"):
            return 1
        elif y == 2018 and dt >= pd.Timestamp("2018-06-30"):
            return 1
        elif y == 2022 and dt >= pd.Timestamp("2022-12-03"):
            return 1
        elif y == 2026 and dt >= pd.Timestamp("2026-07-04"):
            return 1
        return 0

    df["is_knockout"] = df.apply(check_knockout, axis=1)

    matrices_path = config.DATA_PROCESSED / "player_matrices.pkl"
    player_matrices = None
    if matrices_path.exists():
        log.info(f"Loading player matrices from {matrices_path} …")
        with open(matrices_path, "rb") as f:
            player_matrices = pickle.load(f)
        log.info(f"  → {len(player_matrices['home'])} home matrices loaded.")
    else:
        log.warning("Player matrices not found — using zero-filled player features.")

    return df, player_matrices



def get_split_mask(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Temporal split:
      Train: 2014 cycle (defined by config.TRAIN_CYCLES)
      Val:   2018 cycle (defined by config.VAL_CYCLES)
      Test:  2022 cycle (defined by config.TEST_CYCLES)
    We define cycle by the WC they belong to.
    """
    from datetime import timedelta

    WC_DATES = {
        2014: pd.Timestamp("2014-06-12"),
        2018: pd.Timestamp("2018-06-14"),
        2022: pd.Timestamp("2022-11-20"),
    }
    window_days = config.FRIENDLY_WINDOW_MONTHS * 30

    # WC + qualifier: use date year range
    # Friendlies: within window before WC
    def in_cycle(date: pd.Timestamp, wc_year: int) -> bool:
        wc_date = WC_DATES[wc_year]
        cycle_start = WC_DATES.get(wc_year - 4, wc_date - pd.Timedelta(days=4 * 365))
        return cycle_start <= date <= wc_date

    train_mask = df["date"].apply(
        lambda d: any(in_cycle(d, y) for y in config.TRAIN_CYCLES)
    )
    val_mask = df["date"].apply(
        lambda d: any(in_cycle(d, y) for y in config.VAL_CYCLES)
    )
    test_mask = df["date"].apply(
        lambda d: any(in_cycle(d, y) for y in config.TEST_CYCLES)
    )

    log.info(f"Split: {train_mask.sum()} train / {val_mask.sum()} val / {test_mask.sum()} test rows.")
    return train_mask, val_mask, test_mask



def build_context_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the C context feature columns from the Gulati dataset.
    Returns (N, C) float32 array.
    """
    # Use only columns that exist in the dataset
    available = [c for c in config.CONTEXT_FEATURE_COLS if c in df.columns]
    missing   = [c for c in config.CONTEXT_FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning(f"  {len(missing)} context columns not found: {missing[:5]} …")

    context = df[available].fillna(0).values.astype(np.float32)
    return context, available


def build_player_features(df: pd.DataFrame,
                           player_matrices: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """
    For each match in df, retrieve the (11, F) player matrices.
    Returns:
      home_players: (N, 11, F) array
      away_players: (N, 11, F) array
    """
    N = len(df)
    home_arr = np.zeros((N, config.N_PLAYERS, config.F), dtype=np.float32)
    away_arr = np.zeros((N, config.N_PLAYERS, config.F), dtype=np.float32)

    if player_matrices is None:
        log.warning("No player matrices — using zero features for all players.")
        return home_arr, away_arr

    home_mats = player_matrices.get("home", {})
    away_mats = player_matrices.get("away", {})

    matched = 0
    for i, idx in enumerate(df.index):
        if idx in home_mats:
            home_arr[i] = home_mats[idx]
            away_arr[i] = away_mats.get(idx, np.zeros((config.N_PLAYERS, config.F)))
            matched += 1

    log.info(f"  → Player matrices found for {matched}/{N} matches ({100*matched/N:.1f}%).")
    return home_arr, away_arr


def augment_with_swap(home_arr: np.ndarray, away_arr: np.ndarray,
                       context: np.ndarray, targets: np.ndarray,
                       weights: np.ndarray, is_neutral: np.ndarray,
                       context_cols: list[str]) -> tuple:
    """
    Data augmentation: for neutral-venue matches, swap home↔away and invert label.
    Doubles the dataset for neutral-ground matches (most international matches).

    Label mapping: 0 (Home Win) ↔ 2 (Away Win), 1 (Draw) stays.
    """
    log.info("Applying home/away swap augmentation …")

    neutral_mask = is_neutral.astype(bool)
    n_neutral = neutral_mask.sum()

    # Swap home/away player matrices
    aug_home = away_arr[neutral_mask]
    aug_away = home_arr[neutral_mask]

    # Invert labels
    aug_targets = targets[neutral_mask].copy()
    aug_targets[aug_targets == 0] = 99   # temp
    aug_targets[aug_targets == 2] = 0
    aug_targets[aug_targets == 99] = 2

    # Swap context columns: home_xxx ↔ away_xxx
    aug_context = context[neutral_mask].copy()
    for j, col in enumerate(context_cols):
        if col.startswith("home_"):
            away_col = "away_" + col[5:]
            if away_col in context_cols:
                away_j = context_cols.index(away_col)
                aug_context[:, j], aug_context[:, away_j] = (
                    context[neutral_mask, away_j].copy(),
                    context[neutral_mask, j].copy(),
                )
            # Also flip elo_diff, days_since_last_diff, experience_diff, gd diffs, etc.
        elif col in ("elo_diff", "days_since_last_diff", "experience_diff",
                     "win_rate_diff_L5", "gd_avg_diff_L5", "gf_avg_diff_L5",
                     "win_rate_diff_L10", "gd_avg_diff_L10", "gf_avg_diff_L10",
                     "win_rate_diff_L20", "gd_avg_diff_L20", "gf_avg_diff_L20"):
            aug_context[:, j] = -context[neutral_mask, j]
        elif col == "elo_expected_home":
            aug_context[:, j] = 1.0 - context[neutral_mask, j]

    # Augment sample weights
    aug_weights = weights[neutral_mask]

    # Concatenate original + augmented
    all_home    = np.concatenate([home_arr, aug_home],    axis=0)
    all_away    = np.concatenate([away_arr, aug_away],    axis=0)
    all_context = np.concatenate([context, aug_context],  axis=0)
    all_targets = np.concatenate([targets, aug_targets],  axis=0)
    all_weights = np.concatenate([weights, aug_weights],  axis=0)

    log.info(f"  → Added {n_neutral} augmented samples. Total: {len(all_targets)}.")
    return all_home, all_away, all_context, all_targets, all_weights



def fit_scalers(home_train: np.ndarray,
                away_train: np.ndarray,
                context_train: np.ndarray) -> dict:
    """
    Fit StandardScalers on training data.
    Player scaler: fitted on (N*11, F) reshaped data.
    Context scaler: fitted on (N, C) data.
    """
    log.info("Fitting scalers on training data …")

    N, n_p, F = home_train.shape
    player_data = np.concatenate([
        home_train.reshape(-1, F),
        away_train.reshape(-1, F)
    ], axis=0)

    player_scaler = StandardScaler()
    player_scaler.fit(player_data)

    context_scaler = StandardScaler()
    context_scaler.fit(context_train)

    return {"player": player_scaler, "context": context_scaler}


def apply_scalers(home: np.ndarray, away: np.ndarray,
                  context: np.ndarray, scalers: dict):
    """Apply fitted scalers to arrays in-place and clip outlier values."""
    N, n_p, F = home.shape

    home_scaled    = scalers["player"].transform(home.reshape(-1, F)).reshape(N, n_p, F)
    away_scaled    = scalers["player"].transform(away.reshape(-1, F)).reshape(N, n_p, F)
    context_scaled = scalers["context"].transform(context)

    # Item 5: Clip outliers to [-3.0, 3.0] standard deviations to prevent gradient explosion
    home_scaled    = np.clip(home_scaled, -3.0, 3.0)
    away_scaled    = np.clip(away_scaled, -3.0, 3.0)
    context_scaled = np.clip(context_scaled, -3.0, 3.0)

    return home_scaled.astype(np.float32), away_scaled.astype(np.float32), context_scaled.astype(np.float32)



def save_split(path: Path, home: np.ndarray, away: np.ndarray,
               context: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        home_players=home,
        away_players=away,
        context=context,
        targets=targets,
        weights=weights,
    )
    log.info(f"Saved → {path} ({len(targets)} samples)")



def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    df, player_matrices = load_dataset()

    # ── 2. Build context features ─────────────────────────────────────────────
    context_all, context_cols = build_context_features(df)
    log.info(f"Context features: {context_all.shape[1]} columns.")

    # ── 3. Build player matrices ──────────────────────────────────────────────
    home_all, away_all = build_player_features(df, player_matrices)

    # ── 4. Targets & Weights (Item 12) ────────────────────────────────────────
    df["result_encoded"] = df["result"].map(config.RESULT_MAP)
    targets_all = df["result_encoded"].fillna(1).values.astype(np.int64)
    weights_all = df["tourn_weight"].values.astype(np.float32)

    # ── 5. Train / val / test split ───────────────────────────────────────────
    train_mask, val_mask, test_mask = get_split_mask(df)

    train_idx = np.where(train_mask.values)[0]
    val_idx   = np.where(val_mask.values)[0]
    test_idx  = np.where(test_mask.values)[0]

    home_train, away_train = home_all[train_idx], away_all[train_idx]
    home_val,   away_val   = home_all[val_idx],   away_all[val_idx]
    home_test,  away_test  = home_all[test_idx],  away_all[test_idx]
    
    ctx_train, ctx_val, ctx_test = context_all[train_idx], context_all[val_idx], context_all[test_idx]
    tgt_train, tgt_val, tgt_test = targets_all[train_idx], targets_all[val_idx], targets_all[test_idx]
    w_train,   w_val,   w_test   = weights_all[train_idx], weights_all[val_idx], weights_all[test_idx]

    # ── 6. Augmentation (train set only) ─────────────────────────────────────
    is_neutral_train = df["neutral"].values[train_idx]
    home_train, away_train, ctx_train, tgt_train, w_train = augment_with_swap(
        home_train, away_train, ctx_train, tgt_train, w_train,
        is_neutral_train, context_cols
    )

    # ── 7. Fit + apply scalers ────────────────────────────────────────────────
    scalers = fit_scalers(home_train, away_train, ctx_train)
    home_train, away_train, ctx_train = apply_scalers(home_train, away_train, ctx_train, scalers)
    home_val,   away_val,   ctx_val   = apply_scalers(home_val,   away_val,   ctx_val,   scalers)
    home_test,  away_test,  ctx_test  = apply_scalers(home_test,  away_test,  ctx_test,  scalers)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    save_split(config.TRAIN_NPZ, home_train, away_train, ctx_train, tgt_train, w_train)
    save_split(config.VAL_NPZ,   home_val,   away_val,   ctx_val,   tgt_val,   w_val)
    save_split(config.TEST_NPZ,  home_test,  away_test,  ctx_test,  tgt_test,  w_test)


    config.OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    with open(config.SCALER_PKL, "wb") as f:
        pickle.dump({"scalers": scalers, "context_cols": context_cols}, f)
    log.info(f"Saved scalers → {config.SCALER_PKL}")

    # ── 9. Class distribution report ─────────────────────────────────────────
    for name, tgt in [("Train", tgt_train), ("Val", tgt_val), ("Test", tgt_test)]:
        counts = np.bincount(tgt, minlength=3)
        pcts   = counts / counts.sum() * 100
        log.info(f"{name} class distribution: "
                 f"H={counts[0]}({pcts[0]:.1f}%) "
                 f"D={counts[1]}({pcts[1]:.1f}%) "
                 f"A={counts[2]}({pcts[2]:.1f}%)")


    log.info("\n✓ feature_engineering.py complete.")


if __name__ == "__main__":
    main()
