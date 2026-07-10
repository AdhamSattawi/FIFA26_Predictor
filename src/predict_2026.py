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
    
    # Neural networks use only the first 100 context features
    ctx_nn = context[:100] if len(context) > 100 else context
    ctx_t  = torch.from_numpy(ctx_nn).float().unsqueeze(0)         # (1, C_nn)

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


def pick_winner(home_team: str, away_team: str, ens_probs: np.ndarray) -> str:
    """
    In a knockout match the draw is replayed. Redistribute draw probability
    evenly between home and away, then pick the team with higher combined probability.
    """
    p_home, p_draw, p_away = ens_probs
    # Split draw 50/50 between both sides (penalty shootout is coin-flip)
    p_home_ko = p_home + p_draw / 2.0
    p_away_ko = p_away + p_draw / 2.0
    return home_team if p_home_ko >= p_away_ko else away_team


def build_context_for_match(row: pd.Series, context_cols: list,
                             scalers: dict | None,
                             player_mats: dict | None,
                             row_idx: int) -> tuple:
    """Build a scaled context vector for a single fixture row."""
    # Get player matrices
    if player_mats and row_idx in player_mats["home"]:
        hp = player_mats["home"][row_idx]
        ap = player_mats["away"].get(row_idx, np.zeros((config.N_PLAYERS, config.F), dtype=np.float32))
    else:
        hp = np.zeros((config.N_PLAYERS, config.F), dtype=np.float32)
        ap = np.zeros((config.N_PLAYERS, config.F), dtype=np.float32)

    # Squad aggregates
    home_elo   = hp[:, 7].sum()
    away_elo   = ap[:, 7].sum()
    home_goals = hp[:, 0].mean()
    away_goals = ap[:, 0].mean()
    home_assists = hp[:, 1].mean()
    away_assists = ap[:, 1].mean()
    home_age   = hp[:, 4].mean()
    away_age   = ap[:, 4].mean()
    has_l = 1.0 if player_mats and row_idx in player_mats["home"] else 0.0

    squad_extras = {
        "squad_home_elo_sum":        home_elo,
        "squad_away_elo_sum":        away_elo,
        "squad_elo_diff":            home_elo - away_elo,
        "squad_home_goals90_mean":   home_goals,
        "squad_away_goals90_mean":   away_goals,
        "squad_goals90_diff":        home_goals - away_goals,
        "squad_home_assists90_mean": home_assists,
        "squad_away_assists90_mean": away_assists,
        "squad_assists90_diff":      home_assists - away_assists,
        "squad_home_age_mean":       home_age,
        "squad_away_age_mean":       away_age,
        "squad_age_diff":            home_age - away_age,
        "has_lineup":                has_l,
    }
    # Merge squad extras into row for context lookup
    combined = {**{k: row.get(k, 0) for k in context_cols}, **squad_extras}
    ctx_arr = np.array([combined.get(c, 0) for c in context_cols], dtype=np.float32)
    if scalers:
        ctx_arr = scalers["context"].transform(ctx_arr.reshape(1, -1)).flatten().astype(np.float32)
    ctx_arr = np.clip(ctx_arr, -3.0, 3.0)
    return ctx_arr, hp, ap


def run_prediction(home_team: str, away_team: str, date: pd.Timestamp,
                   stage: str, ctx: np.ndarray, hp: np.ndarray, ap: np.ndarray,
                   loaded_nn: dict, xgb_model, ensemble_weights: dict | None,
                   scalers: dict | None) -> tuple:
    """Run all models on one match. Returns (result_row, ensemble_probs)."""
    if scalers:
        hp = scalers["player"].transform(hp.reshape(-1, config.F)).reshape(config.N_PLAYERS, config.F).astype(np.float32)
        ap = scalers["player"].transform(ap.reshape(-1, config.F)).reshape(config.N_PLAYERS, config.F).astype(np.float32)
        hp = np.clip(hp, -3.0, 3.0)
        ap = np.clip(ap, -3.0, 3.0)

    probs = predict_match(loaded_nn, xgb_model, hp, ap, ctx)

    match_row = {
        "date":      date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
        "home_team": home_team,
        "away_team": away_team,
        "stage":     stage,
    }
    for mname, p in probs.items():
        match_row[f"{mname}_home%"] = round(p[0] * 100, 1)
        match_row[f"{mname}_draw%"] = round(p[1] * 100, 1)
        match_row[f"{mname}_away%"] = round(p[2] * 100, 1)

    # Compute ensemble probabilities
    if probs:
        if ensemble_weights is not None:
            name_map = {
                "mlp":       "baseline_mlp",
                "cnn":       "tactical_cnn",
                "attention": "attention_cnn",
                "xgboost":   "xgboost",
            }
            active_weights = {}
            for mkey, pkey in name_map.items():
                if mkey in probs and pkey in ensemble_weights:
                    active_weights[mkey] = ensemble_weights[pkey]
            total_w = sum(active_weights.values())
            if total_w > 0:
                for mkey in active_weights:
                    active_weights[mkey] /= total_w
                ens = np.zeros(3)
                for mkey, w in active_weights.items():
                    ens += probs[mkey] * w
            else:
                ens = np.array(list(probs.values())).mean(axis=0)
        else:
            ens = np.array(list(probs.values())).mean(axis=0)

        match_row["ensemble_home%"] = round(ens[0] * 100, 1)
        match_row["ensemble_draw%"] = round(ens[1] * 100, 1)
        match_row["ensemble_away%"] = round(ens[2] * 100, 1)
        match_row["predicted_outcome"] = config.RESULT_NAMES[ens.argmax()]
    else:
        ens = np.array([1/3, 1/3, 1/3])

    return match_row, ens


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

    # ── 4. Log context metadata ───────────────────────────────────────────────
    sample_ctx_len = len(context_cols)
    C_nn = min(sample_ctx_len, 100)
    log.info(f"Context dimension (XGBoost): {sample_ctx_len} | Context dimension (NNs): {C_nn}")

    # ── 5. Load models ────────────────────────────────────────────────────────
    nn_models = {
        "mlp":       load_trained_model(BaselineMLP,  "baseline_mlp",  C_nn),
        "cnn":       load_trained_model(TacticalCNN,  "tactical_cnn",  C_nn),
        "attention": load_trained_model(AttentionCNN, "attention_cnn", C_nn),
    }
    loaded_nn = {k: v for k, v in nn_models.items() if v is not None}

    xgb_model = None
    xgb_path  = config.OUTPUTS_MODELS / "xgboost_best.pkl"
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        log.info(f"Loaded xgboost from {xgb_path}")

    if not loaded_nn and xgb_model is None:
        log.error("No trained models found. Run src/train.py or src/train_xgb.py first.")
        return

    ensemble_weights = None
    weights_path = config.OUTPUTS_MODELS / "ensemble_weights.pkl"
    if weights_path.exists():
        with open(weights_path, "rb") as f:
            ensemble_weights = pickle.load(f)
        log.info(f"Loaded optimal blend weights: {ensemble_weights}")

    # ── 6. Separate fixtures by stage for dynamic bracket resolution ──────────
    # Sort by date so we process chronologically: QF → SF → Final
    fixtures_sorted = fixtures.sort_values("date").reset_index()
    n = len(fixtures_sorted)

    # Standard bracket structure:
    #   Earliest 4 matches = Quarter-finals
    #   Next 2             = Semi-finals  (teams replaced by QF winners)
    #   Last 1             = Final        (teams replaced by SF winners)
    if n >= 7:
        qf_rows  = fixtures_sorted.iloc[:4]
        sf_rows  = fixtures_sorted.iloc[4:6]
        fin_rows = fixtures_sorted.iloc[6:]
    else:
        qf_rows  = fixtures_sorted
        sf_rows  = pd.DataFrame()
        fin_rows = pd.DataFrame()

    log.info(f"Bracket: {len(qf_rows)} QF(s), {len(sf_rows)} SF(s), {len(fin_rows)} Final(s)")

    results = []

    # ── Inner helper: predict one fixture row, return result + ensemble probs + winner ─
    def predict_fixture_row(fixture_row: pd.Series, stage: str,
                            override_home: str | None = None,
                            override_away: str | None = None):
        row_idx   = fixture_row.get("index", fixture_row.name)
        home_team = override_home or fixture_row["home_team"]
        away_team = override_away or fixture_row["away_team"]
        date      = fixture_row["date"]
        ctx, hp, ap = build_context_for_match(
            fixture_row, context_cols, scalers, player_mats, row_idx
        )
        match_row, ens = run_prediction(
            home_team, away_team, date, stage,
            ctx, hp, ap, loaded_nn, xgb_model, ensemble_weights, scalers
        )
        winner = pick_winner(home_team, away_team, ens)
        return match_row, ens, winner

    # ── 7. Quarter-finals ─────────────────────────────────────────────────────
    log.info("\n── Quarter-finals ──")
    qf_winners = []
    for _, qf_row in qf_rows.iterrows():
        match_row, ens, winner = predict_fixture_row(qf_row, "Quarter-finals")
        results.append(match_row)
        qf_winners.append(winner)
        log.info(
            f"  {match_row['home_team']} vs {match_row['away_team']} "
            f"→ {match_row.get('predicted_outcome', '?')} "
            f"(KO winner: {winner})"
        )

    # ── 8. Semi-finals — teams from QF winners, NOT CSV placeholders ──────────
    sf_winners = []
    if not sf_rows.empty:
        log.info("\n── Semi-finals ──")
        # Bracket pairing: QF0 winner vs QF1 winner = SF1
        #                  QF2 winner vs QF3 winner = SF2
        if len(qf_winners) >= 4:
            sf_matchups = [
                (qf_winners[0], qf_winners[1]),
                (qf_winners[2], qf_winners[3]),
            ]
        elif len(qf_winners) == 2:
            sf_matchups = [(qf_winners[0], qf_winners[1])]
        else:
            sf_matchups = [(r["home_team"], r["away_team"]) for _, r in sf_rows.iterrows()]

        for i, (_, sf_row) in enumerate(sf_rows.iterrows()):
            sf_home, sf_away = sf_matchups[i] if i < len(sf_matchups) else (sf_row["home_team"], sf_row["away_team"])
            match_row, ens, winner = predict_fixture_row(
                sf_row, "Semi-finals",
                override_home=sf_home, override_away=sf_away
            )
            results.append(match_row)
            sf_winners.append(winner)
            log.info(
                f"  {sf_home} vs {sf_away} "
                f"→ {match_row.get('predicted_outcome', '?')} "
                f"(KO winner: {winner})"
            )

    # ── 9. Final — teams from SF winners, NOT CSV placeholders ────────────────
    if not fin_rows.empty:
        log.info("\n── Final ──")
        fin_home = sf_winners[0] if len(sf_winners) > 0 else fin_rows.iloc[0]["home_team"]
        fin_away = sf_winners[1] if len(sf_winners) > 1 else fin_rows.iloc[0]["away_team"]
        match_row, ens, winner = predict_fixture_row(
            fin_rows.iloc[0], "Final",
            override_home=fin_home, override_away=fin_away
        )
        results.append(match_row)
        log.info(
            f"  {fin_home} vs {fin_away} "
            f"→ {match_row.get('predicted_outcome', '?')} "
            f"(🏆 Champion: {winner})"
        )

    # ── 10. Save ───────────────────────────────────────────────────────────────
    out_df  = pd.DataFrame(results)
    out_csv = config.OUTPUTS_PREDICTIONS / "wc2026_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    log.info(f"\n✓ Saved {len(out_df)} match predictions → {out_csv}")

    if "predicted_outcome" in out_df.columns:
        counts = out_df["predicted_outcome"].value_counts()
        log.info(f"\nPrediction distribution:\n{counts.to_string()}")


if __name__ == "__main__":
    main()
