"""
ensemble_2026.py — PlayerElo ensemble for 2026 World Cup predictions.

Option B: A separate lightweight ensemble that blends the base neural network
probabilities with a PlayerElo-derived signal for 2026 only.

Steps:
  1. Load base model predictions (from predict_2026.py output)
  2. Load PlayerElo data (players.csv + coaches.csv)
  3. Load 2026 squad lists
  4. Compute per-team Elo metrics
  5. Derive Elo-based match probabilities
  6. Blend with base model via calibrated α
  7. Save final ensemble predictions

Inputs:
  outputs/predictions/wc2026_predictions.csv   — base model output
  data/raw/player_elo/players.csv              — PlayerElo snapshot
  data/raw/player_elo/coaches.csv              — Coach Elo snapshot

Output:
  outputs/predictions/wc2026_ensemble.csv      — final blended predictions
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Blend weight: α × base_model + (1-α) × elo_model
# α=0.7 means we trust the neural net more; tune this on 2022 backtest
ALPHA = 0.7

# Elo-to-probability scaling factor (standard Elo formula)
ELO_K = 400.0


def load_player_elo() -> pd.DataFrame | None:
    players_path = config.PLAYER_ELO_DIR / "players.csv"
    if not players_path.exists():
        log.warning(f"PlayerElo file not found: {players_path}")
        log.warning("Download from: https://www.kaggle.com/datasets/playerelo/playerelo")
        return None
    df = pd.read_csv(players_path)
    log.info(f"Loaded {len(df)} players from PlayerElo.")
    return df


def load_coach_elo() -> pd.DataFrame | None:
    coaches_path = config.PLAYER_ELO_DIR / "coaches.csv"
    if not coaches_path.exists():
        return None
    df = pd.read_csv(coaches_path)
    log.info(f"Loaded {len(df)} coaches from PlayerElo.")
    return df


def compute_team_elo_features(squad: list[str],
                               player_elo: pd.DataFrame,
                               team_name: str) -> dict:
    """
    Given a list of 11 player names for a team, compute Elo-based team features.

    Returns:
      team_avg_elo: mean Elo of the starting XI
      team_max_elo: max Elo (star power)
      team_elo_std: std dev (squad balance)
      team_ear_180: mean recent form (EAR 180-day) if available
    """
    # Match player names (fuzzy — allow partial match)
    elo_col = "elo"
    ear_col = "ear_180" if "ear_180" in player_elo.columns else None
    name_col = (
        "player_name" if "player_name" in player_elo.columns
        else "name" if "name" in player_elo.columns
        else player_elo.columns[0]
    )

    nat_col = "nationality" if "nationality" in player_elo.columns else None

    # Filter by nationality if available
    if nat_col:
        team_df = player_elo[
            player_elo[nat_col].str.lower().str.contains(
                team_name.lower().split()[0], na=False
            )
        ]
    else:
        team_df = player_elo

    if len(team_df) == 0:
        team_df = player_elo  # fall back to all

    # Try to match squad player names
    matched_elos = []
    matched_ears = []
    for player_name in squad:
        if not player_name or not isinstance(player_name, str):
            continue
        last_name = player_name.strip().split()[-1].lower()
        matches = team_df[team_df[name_col].str.lower().str.contains(last_name, na=False)]
        if len(matches) > 0:
            matched_elos.append(matches[elo_col].max())
            if ear_col:
                matched_ears.append(matches[ear_col].max())

    if len(matched_elos) == 0:
        # Use country-level top players
        if nat_col:
            top_nat = player_elo[
                player_elo[nat_col].str.lower().str.contains(
                    team_name.lower().split()[0], na=False
                )
            ].nlargest(11, elo_col)
            matched_elos = top_nat[elo_col].tolist()
            if ear_col:
                matched_ears = top_nat[ear_col].tolist()

    if not matched_elos:
        # Global median fallback
        median_elo = float(player_elo[elo_col].median())
        matched_elos = [median_elo] * 11

    return {
        "team_avg_elo": float(np.mean(matched_elos)),
        "team_max_elo": float(np.max(matched_elos)),
        "team_elo_std": float(np.std(matched_elos)) if len(matched_elos) > 1 else 0.0,
        "team_ear_180": float(np.mean(matched_ears)) if matched_ears else 0.0,
    }


def elo_to_3class_prob(elo_home: float, elo_away: float) -> np.ndarray:
    """
    Convert Elo difference to 3-class probabilities [Home, Draw, Away].

    Uses the standard Elo expected score formula:
      E_home = 1 / (1 + 10^((elo_away - elo_home) / ELO_K))

    Then splits draw probability using historical base rates:
      P(Draw) ≈ 0.24 (international football average)
      P(Home) = (E_home - 0.24/2) × (1 / (1 - 0.24))
      P(Away) = 1 - P(Home) - P(Draw)
    """
    base_draw_rate = 0.24

    E_home = 1.0 / (1.0 + 10.0 ** ((elo_away - elo_home) / ELO_K))

    # Redistribute draw probability
    p_draw = base_draw_rate
    remaining = 1.0 - p_draw
    p_home = E_home * remaining
    p_away = 1.0 - p_home - p_draw

    # Clip to valid range
    probs = np.array([p_home, p_draw, p_away], dtype=np.float32)
    probs = np.clip(probs, 0.01, 0.98)
    probs /= probs.sum()  # renormalize

    return probs


def blend_probabilities(base_probs: np.ndarray,
                         elo_probs: np.ndarray,
                         alpha: float = ALPHA) -> np.ndarray:
    """Weighted blend: α × base + (1-α) × elo."""
    blended = alpha * base_probs + (1.0 - alpha) * elo_probs
    blended /= blended.sum()
    return blended


def main():
    config.OUTPUTS_PREDICTIONS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load base predictions ──────────────────────────────────────────────
    base_pred_path = config.OUTPUTS_PREDICTIONS / "wc2026_predictions.csv"
    if not base_pred_path.exists():
        log.error(f"Base predictions not found: {base_pred_path}")
        log.error("Run src/predict_2026.py first.")
        return

    base_df = pd.read_csv(base_pred_path)
    log.info(f"Loaded {len(base_df)} base predictions.")

    # ── 2. Load PlayerElo ─────────────────────────────────────────────────────
    player_elo = load_player_elo()
    coach_elo  = load_coach_elo()

    if player_elo is None:
        log.warning("PlayerElo not available. Cannot compute Elo ensemble.")
        log.warning("Copying base predictions as final output.")
        base_df.to_csv(
            config.OUTPUTS_PREDICTIONS / "wc2026_ensemble.csv", index=False
        )
        return

    # ── 3. Compute per-match Elo blend ────────────────────────────────────────
    ensemble_rows = []
    elo_col = "elo"
    name_col = (
        "player_name" if "player_name" in player_elo.columns
        else "name" if "name" in player_elo.columns
        else player_elo.columns[0]
    )

    for _, row in base_df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Get base ensemble probabilities
        if "ensemble_home%" in row:
            base_probs = np.array([
                row["ensemble_home%"] / 100,
                row["ensemble_draw%"] / 100,
                row["ensemble_away%"] / 100,
            ], dtype=np.float32)
        else:
            # Fall back to first available model
            for prefix in ["mlp", "cnn", "attention"]:
                if f"{prefix}_home%" in row:
                    base_probs = np.array([
                        row[f"{prefix}_home%"] / 100,
                        row[f"{prefix}_draw%"] / 100,
                        row[f"{prefix}_away%"] / 100,
                    ], dtype=np.float32)
                    break
            else:
                base_probs = np.array([1/3, 1/3, 1/3], dtype=np.float32)

        # Look up team Elos (using top players by nationality)
        nat_col = "nationality" if "nationality" in player_elo.columns else None

        def get_team_avg_elo(team_name: str) -> float:
            if nat_col is None:
                return float(player_elo[elo_col].median())
            nat_mask = player_elo[nat_col].str.lower().str.contains(
                team_name.lower().split()[0], na=False
            )
            team_players = player_elo[nat_mask]
            if len(team_players) == 0:
                return float(player_elo[elo_col].median())
            return float(team_players.nlargest(11, elo_col)[elo_col].mean())

        elo_home_val = get_team_avg_elo(home_team)
        elo_away_val = get_team_avg_elo(away_team)

        elo_probs = elo_to_3class_prob(elo_home_val, elo_away_val)
        final     = blend_probabilities(base_probs, elo_probs, alpha=ALPHA)

        out_row = dict(row)
        out_row["elo_home_val"]    = round(elo_home_val, 1)
        out_row["elo_away_val"]    = round(elo_away_val, 1)
        out_row["elo_home%"]       = round(elo_probs[0] * 100, 1)
        out_row["elo_draw%"]       = round(elo_probs[1] * 100, 1)
        out_row["elo_away%"]       = round(elo_probs[2] * 100, 1)
        out_row["ensemble_final_home%"] = round(final[0] * 100, 1)
        out_row["ensemble_final_draw%"] = round(final[1] * 100, 1)
        out_row["ensemble_final_away%"] = round(final[2] * 100, 1)
        out_row["final_prediction"] = config.RESULT_NAMES[final.argmax()]
        out_row["elo_diff"]         = round(elo_home_val - elo_away_val, 1)

        # Redistribute draw probability for knockout stage (To Progress)
        total_win = final[0] + final[2]
        prog_home = final[0] / total_win if total_win > 0 else 0.5
        prog_away = final[2] / total_win if total_win > 0 else 0.5
        out_row["to_progress_home%"] = round(prog_home * 100, 1)
        out_row["to_progress_away%"] = round(prog_away * 100, 1)
        out_row["progress_prediction"] = home_team if prog_home > prog_away else away_team

        ensemble_rows.append(out_row)

    # ── 4. Save ───────────────────────────────────────────────────────────────
    out_df  = pd.DataFrame(ensemble_rows)
    out_csv = config.OUTPUTS_PREDICTIONS / "wc2026_ensemble.csv"
    out_df.to_csv(out_csv, index=False)
    log.info(f"\n✓ Saved {len(out_df)} ensemble predictions → {out_csv}")

    # Print predictions with progress probabilities
    log.info("\nEnsemble Predictions & Knockout Progression Probability:")
    print_cols = [
        "home_team", "away_team", "ensemble_final_home%", "ensemble_final_draw%", 
        "ensemble_final_away%", "to_progress_home%", "to_progress_away%", "progress_prediction"
    ]
    print(out_df[print_cols].to_string(index=False))


if __name__ == "__main__":
    main()
