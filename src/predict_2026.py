"""
predict_2026.py — Generate match outcome predictions for the 2026 World Cup.

Steps:
  1. Load 2026 WC fixtures from the Gulati dataset (is_world_cup == 1, date >= 2026)
  2. Load trained models
  3. For each match, build feature matrices using 2026 squad data
  4. Run through all 3 models and generate probability outputs

Note: Player-level features (lineups + club stats) must be scraped first for 2026.
If not yet available, we fall back to zero player matrices (context-only prediction).

Output: outputs/predictions/wc2026_predictions.csv
"""

import sys
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.models.baseline_mlp  import BaselineMLP
from src.models.tactical_cnn  import TacticalCNN
from src.models.attention_cnn import AttentionCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

WC2026_DATE_START = pd.Timestamp("2026-06-11")
WC2026_DATE_END   = pd.Timestamp("2026-07-19")

# Stage ordering for display
STAGE_ORDER = {
    "Group Stage":      0,
    "Round of 32":      1,
    "Round of 16":      2,
    "Quarter-finals":   3,
    "Semi-finals":      4,
    "Third place":      5,
    "Final":            6,
}


def load_2026_fixtures(gulati_path: Path) -> pd.DataFrame:
    """
    Load 2026 WC fixtures from the extended dataset (dataset_with_2026.csv),
    returning only upcoming/unplayed matches (result is NaN).
    Falls back to the original Gulati CSV if the extended file doesn't exist.
    """
    extended_path = config.DATA_PROCESSED / "dataset_with_2026.csv"
    if extended_path.exists():
        df = pd.read_csv(extended_path, parse_dates=["date"])
        log.info(f"Using extended dataset: {extended_path}")
    else:
        df = pd.read_csv(gulati_path, parse_dates=["date"])
        log.warning("Extended dataset not found — using original Gulati CSV. "
                    "Run src/processing/compute_features_2026.py first.")

    wc26 = df[
        (df["is_world_cup"] == 1) &
        (df["date"] >= WC2026_DATE_START)
    ].copy()

    # Only predict matches where result is not yet known
    upcoming = wc26[wc26["result"].isna()].copy()

    if len(upcoming) == 0:
        log.warning(
            "No unplayed 2026 WC fixtures found. "
            "Using the most recent 50 WC matches as a demonstration."
        )
        wc = df[df["is_world_cup"] == 1].tail(50).copy()
        return wc

    log.info(f"Loaded {len(upcoming)} upcoming 2026 WC fixtures to predict.")
    return upcoming


def load_2026_player_matrices(lineups_path: Path, player_stats_path: Path) -> dict | None:
    """
    Load 2026-specific player matrices if available.
    Returns None if scraping hasn't been done yet.
    """
    if not lineups_path.exists() or not player_stats_path.exists():
        log.warning("2026 lineup/player stats not found — using zero player features.")
        return None

    # Re-use merge_data logic inline for 2026 cycle
    try:
        from src.processing.merge_data import (
            compute_position_medians, build_player_matrix, join_lineups_to_gulati
        )
        lineups   = pd.read_csv(lineups_path, parse_dates=["match_date"])
        p_stats   = pd.read_csv(player_stats_path)
        p_stats["player_id"] = p_stats["player_id"].astype(str)
        # Filter to 2026 cycle only
        lineups = lineups[lineups["match_date"] >= WC2026_DATE_START]
        medians = compute_position_medians(p_stats, lineups)
        # Build a dummy gulati DataFrame for joining
        fixtures_df = load_2026_fixtures(config.GULATI_CSV)
        _, lineups_matched = join_lineups_to_gulati(fixtures_df, lineups)
        home_mats, away_mats = {}, {}
        for match_idx, group in lineups_matched.groupby("match_idx"):
            row = fixtures_df[fixtures_df.index == match_idx]
            if not len(row):
                continue
            home_team = row.iloc[0]["home_team"]
            away_team = row.iloc[0]["away_team"]
            from src.scraping.utils import normalize_team_name
            teams = {normalize_team_name(t).lower(): t for t in group["team"].unique()}
            ht = teams.get(normalize_team_name(home_team).lower())
            at = teams.get(normalize_team_name(away_team).lower())
            if ht and at:
                home_mats[match_idx] = build_player_matrix(group, p_stats, ht, 2026, medians)
                away_mats[match_idx] = build_player_matrix(group, p_stats, at, 2026, medians)
        return {"home": home_mats, "away": away_mats}
    except Exception as e:
        log.warning(f"Could not build 2026 player matrices: {e}")
        return None


def load_scalers() -> dict | None:
    if not config.SCALER_PKL.exists():
        log.warning("Scalers not found — features will not be normalized.")
        return None
    with open(config.SCALER_PKL, "rb") as f:
        return pickle.load(f)


def load_trained_model(model_class, name: str, C: int) -> torch.nn.Module | None:
    path = config.OUTPUTS_MODELS / f"{name}_best.pt"
    if not path.exists():
        log.warning(f"  {name} checkpoint not found: {path}")
        return None
    model = model_class(F=config.F, C=C)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def predict_match(models: dict, xgb_model, home_players: np.ndarray,
                  away_players: np.ndarray, context: np.ndarray) -> dict:
    """Run all models on a single match. Returns dict of probabilities."""
    home_t = torch.from_numpy(home_players).float().unsqueeze(0)   # (1, 11, F)
    away_t = torch.from_numpy(away_players).float().unsqueeze(0)
    ctx_t  = torch.from_numpy(context).float().unsqueeze(0)         # (1, C)

    probs = {}
    for model_name, model in models.items():
        if model is None:
            continue
        logits = model(home_t, away_t, ctx_t)
        p      = torch.softmax(logits, dim=1).squeeze(0).numpy()
        probs[model_name] = p

    if xgb_model is not None:
        p = xgb_model.predict_proba(context.reshape(1, -1))[0]
        probs["xgboost"] = p

    return probs


def infer_stage(tournament_str: str, date: pd.Timestamp) -> str:
    """Infer match stage from tournament string."""
    t = str(tournament_str).lower()
    if "final" in t and "quarter" not in t and "semi" not in t and "third" not in t:
        return "Final"
    if "semi" in t:
        return "Semi-finals"
    if "quarter" in t:
        return "Quarter-finals"
    if "third" in t or "3rd" in t:
        return "Third place"
    if "round of 16" in t or "r16" in t:
        return "Round of 16"
    if "round of 32" in t or "r32" in t:
        return "Round of 32"
    return "Group Stage"


def main():
    config.OUTPUTS_PREDICTIONS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load fixtures ──────────────────────────────────────────────────────
    fixtures = load_2026_fixtures(config.GULATI_CSV)
    log.info(f"Processing {len(fixtures)} 2026 matches …")

    # ── 2. Load scalers ───────────────────────────────────────────────────────
    scaler_data = load_scalers()
    if scaler_data:
        scalers      = scaler_data["scalers"]
        context_cols = scaler_data["context_cols"]
    else:
        scalers      = None
        context_cols = config.CONTEXT_FEATURE_COLS

    # ── 3. Load player matrices ───────────────────────────────────────────────
    player_mats = load_2026_player_matrices(
        config.LINEUPS_CSV, config.PLAYER_STATS_CSV
    )

    # ── 4. Build context array ────────────────────────────────────────────────
    # Compute squad aggregates for upcoming 2026 fixtures
    log.info("Computing squad aggregates for 2026 prediction fixtures...")
    squad_data = []
    for row_idx, row in fixtures.iterrows():
        if player_mats and row_idx in player_mats["home"]:
            hp = player_mats["home"][row_idx]
            ap = player_mats["away"].get(row_idx, np.zeros((config.N_PLAYERS, config.F), dtype=np.float32))
        else:
            hp = np.zeros((config.N_PLAYERS, config.F), dtype=np.float32)
            ap = np.zeros((config.N_PLAYERS, config.F), dtype=np.float32)
            
        home_elo = hp[:, 7].sum()
        away_elo = ap[:, 7].sum()
        elo_diff = home_elo - away_elo

        home_goals = hp[:, 0].mean()
        away_goals = ap[:, 0].mean()
        goals_diff = home_goals - away_goals

        home_assists = hp[:, 1].mean()
        away_assists = ap[:, 1].mean()
        assists_diff = home_assists - away_assists

        home_age = hp[:, 4].mean()
        away_age = ap[:, 4].mean()
        age_diff = home_age - away_age

        squad_data.append([
            home_elo, away_elo, elo_diff,
            home_goals, away_goals, goals_diff,
            home_assists, away_assists, assists_diff,
            home_age, away_age, age_diff
        ])
    
    squad_cols = [
        "squad_home_elo_sum", "squad_away_elo_sum", "squad_elo_diff",
        "squad_home_goals90_mean", "squad_away_goals90_mean", "squad_goals90_diff",
        "squad_home_assists90_mean", "squad_away_assists90_mean", "squad_assists90_diff",
        "squad_home_age_mean", "squad_away_age_mean", "squad_age_diff"
    ]
    squad_df = pd.DataFrame(squad_data, columns=squad_cols, index=fixtures.index)
    fixtures = pd.concat([fixtures, squad_df], axis=1)

    available_cols = [c for c in context_cols if c in fixtures.columns]
    ctx_array = fixtures[available_cols].fillna(0).values.astype(np.float32)
    if scalers:
        ctx_array = scalers["context"].transform(ctx_array).astype(np.float32)
    C_actual = ctx_array.shape[1]
    log.info(f"Context dimension: {C_actual}")

    # ── 5. Load models ────────────────────────────────────────────────────────
    models = {
        "mlp":       load_trained_model(BaselineMLP,  "baseline_mlp",  C_actual),
        "cnn":       load_trained_model(TacticalCNN,  "tactical_cnn",  C_actual),
        "attention": load_trained_model(AttentionCNN, "attention_cnn", C_actual),
    }
    loaded = {k: v for k, v in models.items() if v is not None}
    
    # Load XGBoost model if it exists
    xgb_model = None
    xgb_path = config.OUTPUTS_MODELS / "xgboost_best.pkl"
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        log.info(f"Loaded xgboost from {xgb_path}")

    if not loaded and xgb_model is None:
        log.error("No trained models found. Run src/train.py or src/train_xgb.py first.")
        return

    # Load optimal ensemble weights (Item 7)
    ensemble_weights = None
    weights_path = config.OUTPUTS_MODELS / "ensemble_weights.pkl"
    if weights_path.exists():
        with open(weights_path, "rb") as f:
            ensemble_weights = pickle.load(f)
        log.info(f"Loaded optimal blend weights: {ensemble_weights}")



    # ── 6. Generate predictions ───────────────────────────────────────────────
    results = []
    home_mats = player_mats["home"] if player_mats else {}
    away_mats = player_mats["away"] if player_mats else {}

    for i, (row_idx, row) in enumerate(fixtures.iterrows()):
        home_team = row["home_team"]
        away_team = row["away_team"]
        date      = row["date"]

        # Player matrices (zero if not available)
        if row_idx in home_mats:
            hp = home_mats[row_idx]
            ap = away_mats.get(row_idx, np.zeros((config.N_PLAYERS, config.F), dtype=np.float32))
        else:
            hp = np.zeros((config.N_PLAYERS, config.F), dtype=np.float32)
            ap = np.zeros((config.N_PLAYERS, config.F), dtype=np.float32)

        # Apply scaling and clipping to prediction features (Item 5)
        if scalers:
            hp = scalers["player"].transform(hp.reshape(-1, config.F)).reshape(config.N_PLAYERS, config.F).astype(np.float32)
            ap = scalers["player"].transform(ap.reshape(-1, config.F)).reshape(config.N_PLAYERS, config.F).astype(np.float32)
            hp = np.clip(hp, -3.0, 3.0)
            ap = np.clip(ap, -3.0, 3.0)
        
        ctx = ctx_array[i]
        ctx = np.clip(ctx, -3.0, 3.0)

        probs = predict_match(loaded, xgb_model, hp, ap, ctx)


        # Build output row
        match_row = {
            "date":      date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
            "home_team": home_team,
            "away_team": away_team,
            "stage":     infer_stage(row.get("tournament", ""), date),
        }

        # Individual model probabilities
        for mname, p in probs.items():
            match_row[f"{mname}_home%"] = round(p[0] * 100, 1)
            match_row[f"{mname}_draw%"] = round(p[1] * 100, 1)
            match_row[f"{mname}_away%"] = round(p[2] * 100, 1)

        # Ensemble: weighted average of available models if weights exist, else simple average (Item 7)
        if probs:
            if ensemble_weights is not None:
                name_map = {
                    "mlp":       "baseline_mlp",
                    "cnn":       "tactical_cnn",
                    "attention": "attention_cnn",
                    "xgboost":   "xgboost",
                }
                # Filter weights to active models
                active_weights = {}
                for mkey, pkey in name_map.items():
                    if mkey in probs and pkey in ensemble_weights:
                        active_weights[mkey] = ensemble_weights[pkey]
                
                # Renormalize active weights to sum to 1
                total_w = sum(active_weights.values())
                if total_w > 0:
                    for mkey in active_weights:
                        active_weights[mkey] /= total_w
                    
                    # Compute weighted combination
                    ens = np.zeros(3)
                    for mkey, w in active_weights.items():
                        ens += probs[mkey] * w
                else:
                    ens = np.array(list(probs.values())).mean(axis=0)
            else:
                prob_stack = np.array(list(probs.values()))
                ens = prob_stack.mean(axis=0)

            match_row["ensemble_home%"] = round(ens[0] * 100, 1)
            match_row["ensemble_draw%"] = round(ens[1] * 100, 1)
            match_row["ensemble_away%"] = round(ens[2] * 100, 1)
            match_row["predicted_outcome"] = config.RESULT_NAMES[ens.argmax()]


        results.append(match_row)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out_df  = pd.DataFrame(results)
    out_csv = config.OUTPUTS_PREDICTIONS / "wc2026_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    log.info(f"\n✓ Saved {len(out_df)} match predictions → {out_csv}")

    # Print summary
    if "predicted_outcome" in out_df.columns:
        counts = out_df["predicted_outcome"].value_counts()
        log.info(f"\nPrediction distribution:\n{counts.to_string()}")


if __name__ == "__main__":
    main()
