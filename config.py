"""
config.py — Central configuration for FIFA 26 Predictor.
All paths, constants, feature lists, and hyperparameters live here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

DATA_RAW          = ROOT / "data" / "raw"
DATA_PROCESSED    = ROOT / "data" / "processed"
DATA_FEATURES     = ROOT / "data" / "features"

GULATI_DIR        = DATA_RAW / "gulati_dataset"
LINEUPS_DIR       = DATA_RAW / "lineups"
PLAYER_STATS_DIR  = DATA_RAW / "player_stats"
PLAYER_ELO_DIR    = DATA_RAW / "player_elo"

OUTPUTS_MODELS      = ROOT / "outputs" / "models"
OUTPUTS_PLOTS       = ROOT / "outputs" / "plots"
OUTPUTS_PREDICTIONS = ROOT / "outputs" / "predictions"

# Main dataset (added directly to project root)
GULATI_CSV = ROOT / "world_cup_features_dataset.csv"

# Scraped data outputs
LINEUPS_CSV      = LINEUPS_DIR / "all_lineups.csv"
PLAYER_STATS_CSV = PLAYER_STATS_DIR / "all_player_stats.csv"

# Processed outputs
FULL_DATASET_CSV = DATA_PROCESSED / "full_dataset.csv"
TRAIN_NPZ        = DATA_FEATURES / "train_features.npz"
VAL_NPZ          = DATA_FEATURES / "val_features.npz"
TEST_NPZ         = DATA_FEATURES / "test_features.npz"
SCALER_PKL       = OUTPUTS_MODELS / "scalers.pkl"

# ── Transfermarkt Competition IDs ─────────────────────────────────────────────
WC_COMPETITIONS = {
    "WC2014": {"id": "WM14", "saison": 2013, "year": 2014},
    "WC2018": {"id": "WM18", "saison": 2017, "year": 2018},
    "WC2022": {"id": "WM22", "saison": 2022, "year": 2022},
}

QUALIFIER_IDS = {
    "UEFA":      "WMQE",
    "CONMEBOL":  "SA1Q",
    "CONCACAF":  "CONC",
    "AFC":       "AS1Q",
    "CAF":       "AF1Q",
    "OFC":       "OC1Q",
    "Playoffs":  "WMQ2",
}

FRIENDLY_ID = "ISL"  # International friendlies

# Club season to scrape player stats from per WC cycle
# WC year → Transfermarkt saison ID
SEASON_MAPPING = {
    2014: "2013",   # WC Jun 2014  → 2013-14 club season
    2018: "2017",   # WC Jun 2018  → 2017-18 club season
    2022: "2021",   # WC Nov 2022  → 2021-22 club season (full season before)
    2026: "2025",   # WC Jun 2026  → 2025-26 club season
}

# ── Dataset column groups ─────────────────────────────────────────────────────
# Identifier / metadata columns (not used as model features)
META_COLS = [
    "date", "home_team", "away_team",
    "home_score", "away_score", "result",
    "tournament", "is_world_cup", "is_qualifier", "is_friendly",
    "home_confed", "away_confed",
]

# Label encoding for result
RESULT_MAP = {"H": 0, "D": 1, "A": 2}
RESULT_NAMES = ["Home Win", "Draw", "Away Win"]

# All 102 contextual feature columns from the Gulati dataset
# (everything except META_COLS)
CONTEXT_FEATURE_COLS = [
    # Elo (4)
    "elo_home", "elo_away", "elo_diff", "elo_expected_home",
    # Match context (8)
    "neutral", "tourn_weight", "same_confederation",
    "true_home_advantage", "is_knockout",
    # Fatigue & experience (6)
    "home_days_since_last", "away_days_since_last", "days_since_last_diff",
    "home_total_matches", "away_total_matches", "experience_diff",
    # Rolling form L5 (11 home + 11 away + 3 diff = 25)
    "home_matches_L5", "home_win_rate_L5", "home_draw_rate_L5", "home_loss_rate_L5",
    "home_gf_avg_L5", "home_ga_avg_L5", "home_gd_avg_L5",
    "home_clean_sheet_rate_L5", "home_btts_rate_L5", "home_win_streak_L5", "home_scoring_rate_L5",
    "away_matches_L5", "away_win_rate_L5", "away_draw_rate_L5", "away_loss_rate_L5",
    "away_gf_avg_L5", "away_ga_avg_L5", "away_gd_avg_L5",
    "away_clean_sheet_rate_L5", "away_btts_rate_L5", "away_win_streak_L5", "away_scoring_rate_L5",
    "win_rate_diff_L5", "gd_avg_diff_L5", "gf_avg_diff_L5",
    # Rolling form L10 (25)
    "home_matches_L10", "home_win_rate_L10", "home_draw_rate_L10", "home_loss_rate_L10",
    "home_gf_avg_L10", "home_ga_avg_L10", "home_gd_avg_L10",
    "home_clean_sheet_rate_L10", "home_btts_rate_L10", "home_win_streak_L10", "home_scoring_rate_L10",
    "away_matches_L10", "away_win_rate_L10", "away_draw_rate_L10", "away_loss_rate_L10",
    "away_gf_avg_L10", "away_ga_avg_L10", "away_gd_avg_L10",
    "away_clean_sheet_rate_L10", "away_btts_rate_L10", "away_win_streak_L10", "away_scoring_rate_L10",
    "win_rate_diff_L10", "gd_avg_diff_L10", "gf_avg_diff_L10",
    # Rolling form L20 (25)
    "home_matches_L20", "home_win_rate_L20", "home_draw_rate_L20", "home_loss_rate_L20",
    "home_gf_avg_L20", "home_ga_avg_L20", "home_gd_avg_L20",
    "home_clean_sheet_rate_L20", "home_btts_rate_L20", "home_win_streak_L20", "home_scoring_rate_L20",
    "away_matches_L20", "away_win_rate_L20", "away_draw_rate_L20", "away_loss_rate_L20",
    "away_gf_avg_L20", "away_ga_avg_L20", "away_gd_avg_L20",
    "away_clean_sheet_rate_L20", "away_btts_rate_L20", "away_win_streak_L20", "away_scoring_rate_L20",
    "win_rate_diff_L20", "gd_avg_diff_L20", "gf_avg_diff_L20",
    # Head-to-head (4)
    "h2h_matches", "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate",
    # Penalty & shootout (8)
    "home_penalty_reliance", "away_penalty_reliance",
    "home_owngoal_benefit_rate", "away_owngoal_benefit_rate",
    "home_shootout_win_rate", "away_shootout_win_rate",
]

# C = number of context features
C = len(CONTEXT_FEATURE_COLS)  # 84 actual numeric context cols

# ── Player feature names (F = 8) ──────────────────────────────────────────────
PLAYER_FEATURES = [
    "goals_per90",
    "assists_per90",
    "minutes_pct",
    "appearances_norm",
    "age_norm",
    "goals_norm",
    "assists_norm",
    "discipline",
]
F = len(PLAYER_FEATURES)  # 8

# ── Canonical position ordering (11 slots: GK → ST) ──────────────────────────
POSITION_ORDER = [
    "GK",   # 0 — Goalkeeper
    "RB",   # 1 — Right back / Right wing-back
    "CB1",  # 2 — Centre back (right)
    "CB2",  # 3 — Centre back (left)
    "LB",   # 4 — Left back / Left wing-back
    "CDM",  # 5 — Defensive midfielder
    "CM",   # 6 — Central midfielder
    "CAM",  # 7 — Attacking midfielder
    "RW",   # 8 — Right winger / Right midfielder
    "LW",   # 9 — Left winger / Left midfielder
    "ST",   # 10 — Striker / Centre forward
]
N_PLAYERS = len(POSITION_ORDER)  # 11

# ── Training / validation / testing split ───────────────────────────────────────
# Temporal: train on 2014+2018, test on 2022.
# Validation is a 15% random holdout from the training pool (not a full cycle).
TRAIN_CYCLES = [2014, 2018]
VAL_CYCLES   = []          # unused — validation done via VAL_HOLDOUT_FRAC
TEST_CYCLES  = [2022]
VAL_HOLDOUT_FRAC = 0.15   # fraction of training pool held out for early stopping
# Friendly window: only include friendlies within 12 months before each WC
FRIENDLY_WINDOW_MONTHS = 12

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE           = 32
LEARNING_RATE        = 1e-3
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 25    # Model 1 (MLP)
CNN_PATIENCE         = 35    # Model 2 (Tactical CNN)
ATT_PATIENCE         = 40    # Model 3 (Attention CNN)
SEED                 = 42

