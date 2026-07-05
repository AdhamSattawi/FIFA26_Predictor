"""
merge_data.py — Join the Gulati dataset, scraped lineups, and player stats
into one unified dataset ready for feature engineering.

Outputs:
  data/processed/full_dataset.csv     — one row per match, all context features + player refs
  data/processed/player_matrices.pkl  — dict: match_idx → (home_matrix, away_matrix)
"""

import sys
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_gulati(filter_scope: bool = True) -> pd.DataFrame:
    """
    Load the Gulati dataset and optionally filter to our training scope:
      - WC matches (is_world_cup == 1)
      - Qualifier matches (is_qualifier == 1)
      - Friendlies within 12-month window before each WC
    """
    log.info(f"Loading Gulati dataset from {config.GULATI_CSV} …")
    df = pd.read_csv(config.GULATI_CSV, parse_dates=["date"])
    log.info(f"  → {len(df)} total rows loaded.")

    if not filter_scope:
        return df

    from datetime import timedelta
    WC_DATES = {
        2014: pd.Timestamp("2014-06-12"),
        2018: pd.Timestamp("2018-06-14"),
        2022: pd.Timestamp("2022-11-20"),
        2026: pd.Timestamp("2026-06-11"),
    }
    window_days = config.FRIENDLY_WINDOW_MONTHS * 30

    mask_wc   = df["is_world_cup"] == 1
    mask_qual = df["is_qualifier"] == 1

    friendly_masks = [
        (df["is_friendly"] == 1)
        & (df["date"] >= (wc_date - timedelta(days=window_days)))
        & (df["date"] <= wc_date)
        for wc_date in WC_DATES.values()
    ]
    import functools, operator
    mask_friendly = functools.reduce(operator.or_, friendly_masks)

    df_scope = df[mask_wc | mask_qual | mask_friendly].copy().reset_index(drop=True)
    log.info(f"  → {len(df_scope)} rows after scope filter "
             f"({mask_wc.sum()} WC, {mask_qual.sum()} qual, {mask_friendly.sum()} friendly).")
    return df_scope


def load_lineups() -> pd.DataFrame | None:
    if not config.LINEUPS_CSV.exists():
        log.warning(f"Lineups file not found: {config.LINEUPS_CSV}. "
                    "Run scrape_lineups.py first.")
        return None
    df = pd.read_csv(config.LINEUPS_CSV, parse_dates=["match_date"])
    log.info(f"Loaded {len(df)} lineup rows from {config.LINEUPS_CSV}.")
    return df


def load_player_stats() -> pd.DataFrame | None:
    if not config.PLAYER_STATS_CSV.exists():
        log.warning(f"Player stats file not found: {config.PLAYER_STATS_CSV}. "
                    "Run scrape_player_stats.py first.")
        return None
    df = pd.read_csv(config.PLAYER_STATS_CSV)
    df["player_id"] = df["player_id"].astype(str)
    log.info(f"Loaded {len(df)} player stat rows from {config.PLAYER_STATS_CSV}.")
    return df


def join_lineups_to_gulati(gulati: pd.DataFrame,
                            lineups: pd.DataFrame) -> pd.DataFrame:
    """
    Match each Gulati row to lineup rows using (date ± 1 day, home_team, away_team).
    Adds a 'match_idx' column to lineups for downstream merging.
    """
    from src.scraping.utils import normalize_team_name

    log.info("Joining lineups to Gulati dataset …")

    # Normalize team names in Gulati
    gulati = gulati.copy()
    gulati["home_team_norm"] = gulati["home_team"].apply(normalize_team_name)
    gulati["away_team_norm"] = gulati["away_team"].apply(normalize_team_name)
    gulati["match_idx"] = gulati.index

    lineups = lineups.copy()
    lineups["home_team_norm"] = lineups["home_team"].apply(normalize_team_name)
    lineups["away_team_norm"] = lineups["away_team"].apply(normalize_team_name)

    # Build a lookup: (date_str, home_norm, away_norm) → match_idx
    # Allow ±1 day tolerance
    lookup: dict[tuple, int] = {}
    for _, row in gulati.iterrows():
        for delta in range(-1, 2):  # -1, 0, +1 days
            shifted = row["date"] + pd.Timedelta(days=delta)
            key = (shifted.strftime("%Y-%m-%d"),
                   row["home_team_norm"].lower(),
                   row["away_team_norm"].lower())
            lookup[key] = int(row["match_idx"])

    def find_idx(row):
        key = (row["match_date"].strftime("%Y-%m-%d"),
               row["home_team_norm"].lower(),
               row["away_team_norm"].lower())
        return lookup.get(key, -1)

    if lineups.empty:
        lineups["match_idx"] = pd.Series(dtype=int)
    else:
        lineups["match_idx"] = lineups.apply(find_idx, axis=1)
    matched = lineups[lineups["match_idx"] >= 0]
    unmatched = lineups[lineups["match_idx"] < 0]

    log.info(f"  → {len(matched)} lineup rows matched ({len(unmatched)} unmatched).")
    if len(unmatched) > 0:
        log.warning(f"  Unmatched lineup rows (first 5):")
        for _, r in unmatched.head(5).iterrows():
            log.warning(f"    {r['match_date'].date()} {r['home_team']} vs {r['away_team']}")

    return gulati, matched


def build_player_matrix(match_lineups: pd.DataFrame,
                         player_stats: pd.DataFrame,
                         team: str,
                         wc_cycle: int,
                         position_medians: dict) -> np.ndarray:
    """
    Build an (11, F) player feature matrix for one team in one match.
    Missing stats → filled with position-group medians.

    Returns np.ndarray of shape (N_PLAYERS, F).
    """
    from src.processing.position_mapping import lineup_to_slots, map_position

    # Get canonical slots for this team
    team_lineups = match_lineups[match_lineups["team"] == team]
    slots = lineup_to_slots(team_lineups, team)

    saison = config.SEASON_MAPPING.get(wc_cycle, str(wc_cycle - 1))

    matrix = np.zeros((config.N_PLAYERS, config.F), dtype=np.float32)

    for i, slot in enumerate(slots):
        pid = str(slot.get("player_id") or "")
        pos = slot.get("canonical_slot", "CM")

        # Look up player stats
        row = player_stats[
            (player_stats["player_id"] == pid) &
            (player_stats["saison"] == saison)
        ]

        if len(row) > 0:
            r = row.iloc[0]
            apps     = float(r.get("appearances", 0) or 0)
            goals    = float(r.get("goals", 0) or 0)
            assists  = float(r.get("assists", 0) or 0)
            minutes  = float(r.get("minutes_played", 0) or 0)
            yellows  = float(r.get("yellow_cards", 0) or 0)
            reds     = float(r.get("red_cards", 0) or 0)

            # Age calculation
            dob_str = str(r.get("date_of_birth", ""))
            try:
                dob = pd.to_datetime(dob_str, dayfirst=True, errors="coerce")
                wc_date = pd.Timestamp(f"{wc_cycle}-06-15")
                age = (wc_date - dob).days / 365.25 if pd.notna(dob) else 27.0
            except Exception:
                age = 27.0

            # Feature vector (raw — will be normalized later by feature_engineering.py)
            matrix[i] = [
                (goals / minutes * 90) if minutes > 0 else 0.0,   # goals_per90
                (assists / minutes * 90) if minutes > 0 else 0.0,  # assists_per90
                min(minutes / (38 * 90), 1.0),                     # minutes_pct (vs full season)
                apps,                                               # appearances_norm (raw)
                age,                                                # age_norm (raw)
                goals,                                              # goals_norm (raw)
                assists,                                            # assists_norm (raw)
                ((yellows + 3 * reds) / apps) if apps > 0 else 0,  # discipline
            ]
        else:
            # Use position-group median
            median = position_medians.get(pos, position_medians.get("CM", np.zeros(config.F)))
            matrix[i] = median

    return matrix


def compute_position_medians(player_stats: pd.DataFrame) -> dict:
    """
    Compute per-position median feature values as fallback for missing players.
    (Very basic — uses all players regardless of position for now.)
    """
    log.info("Computing position fallback medians …")
    # Use global medians as a simple fallback
    global_median = np.zeros(config.F, dtype=np.float32)
    if len(player_stats) > 0:
        stats = player_stats.copy()
        stats["apps"] = stats["appearances"].fillna(0).clip(lower=1)
        global_median[0] = (stats["goals"] / stats["minutes_played"] * 90).replace([np.inf, -np.inf], 0).median()
        global_median[1] = (stats["assists"] / stats["minutes_played"] * 90).replace([np.inf, -np.inf], 0).median()
        global_median[2] = (stats["minutes_played"] / (38 * 90)).clip(0, 1).median()
        global_median[3] = stats["appearances"].median()
        global_median[4] = 27.0  # average age
        global_median[5] = stats["goals"].median()
        global_median[6] = stats["assists"].median()
        global_median[7] = ((stats["yellow_cards"] + 3 * stats["red_cards"]) / stats["apps"]).replace([np.inf, -np.inf], 0).median()

    # Return same median for all positions (can be made position-specific later)
    return {pos: global_median.copy() for pos in config.POSITION_ORDER + ["GK", "CB", "CB1", "CB2"]}


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    gulati   = load_gulati(filter_scope=True)
    lineups  = load_lineups()
    p_stats  = load_player_stats()

    if lineups is None or p_stats is None:
        log.warning("\n⚠ Lineups or player stats missing — cannot build full player matrices.")
        log.warning("  Saving Gulati context-only dataset (player matrices will be empty).")

        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        gulati["result_encoded"] = gulati["result"].map(config.RESULT_MAP)
        gulati.to_csv(config.FULL_DATASET_CSV, index=False)
        log.info(f"  Saved context-only dataset → {config.FULL_DATASET_CSV}")
        return

    # ── 2. Join lineups to Gulati ─────────────────────────────────────────────
    gulati, lineups_matched = join_lineups_to_gulati(gulati, lineups)

    # ── 3. Compute position medians (for missing player fallback) ─────────────
    position_medians = compute_position_medians(p_stats)

    # ── 4. Build player matrices per match ────────────────────────────────────
    log.info("Building player matrices …")
    home_matrices: dict[int, np.ndarray] = {}
    away_matrices: dict[int, np.ndarray] = {}

    # Determine WC cycle per lineup row
    lineups_matched = lineups_matched.copy()
    lineups_matched["wc_cycle"] = lineups_matched["match_date"].dt.year.map(
        lambda y: 2014 if 2012 <= y <= 2014
             else 2018 if 2016 <= y <= 2018
             else 2022 if 2020 <= y <= 2022
             else 2026 if 2024 <= y <= 2026
             else None
    )

    for match_idx, group in lineups_matched.groupby("match_idx"):
        wc_cycle = group["wc_cycle"].iloc[0]
        if pd.isna(wc_cycle):
            continue
        wc_cycle = int(wc_cycle)

        teams = group["team"].unique()
        if len(teams) < 2:
            continue

        gulati_row = gulati[gulati.index == match_idx]
        if len(gulati_row) == 0:
            continue
        home_team = gulati_row.iloc[0]["home_team"]
        away_team = gulati_row.iloc[0]["away_team"]

        from src.scraping.utils import normalize_team_name
        # Match teams (normalize)
        home_norm = normalize_team_name(home_team).lower()
        away_norm = normalize_team_name(away_team).lower()

        # Find which scraped team names correspond to home/away
        scraped_teams_norm = {normalize_team_name(t).lower(): t for t in teams}

        home_scraped = scraped_teams_norm.get(home_norm)
        away_scraped = scraped_teams_norm.get(away_norm)

        if home_scraped and away_scraped:
            home_matrices[match_idx] = build_player_matrix(
                group, p_stats, home_scraped, wc_cycle, position_medians
            )
            away_matrices[match_idx] = build_player_matrix(
                group, p_stats, away_scraped, wc_cycle, position_medians
            )

    log.info(f"  → Built matrices for {len(home_matrices)} matches.")

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    gulati["result_encoded"] = gulati["result"].map(config.RESULT_MAP)
    gulati.to_csv(config.FULL_DATASET_CSV, index=False)
    log.info(f"Saved full dataset → {config.FULL_DATASET_CSV}")

    matrices_path = config.DATA_PROCESSED / "player_matrices.pkl"
    with open(matrices_path, "wb") as f:
        pickle.dump({"home": home_matrices, "away": away_matrices}, f)
    log.info(f"Saved player matrices → {matrices_path}")
    log.info("\n✓ merge_data.py complete.")


if __name__ == "__main__":
    main()
