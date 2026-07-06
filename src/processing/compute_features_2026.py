"""
compute_features_2026.py
Extends the Gulati dataset with fully-featured rows for 2026 international matches.

Steps:
  1. Load Gulati dataset (ends Dec 31 2025) to get last-known state per team
  2. Load 2026 raw matches from fetch_2026_matches.py output
  3. Process each 2026 match chronologically:
       - Compute Elo update
       - Compute rolling form L5 / L10 / L20
       - Compute H2H record
       - Compute fatigue / experience
  4. Build a new feature row (same 102 columns as Gulati) for each 2026 match
  5. Save to data/processed/dataset_with_2026.csv

Note: Models are NOT retrained. This script only builds the feature matrix
      for prediction input. Rows with NaN scores are UPCOMING matches.
"""

import sys
import math
import collections
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

# ── K-factor mapping by tournament weight ─────────────────────────────────────
TOURN_WEIGHTS = {
    "FIFA World Cup":                   8.0,
    "FIFA World Cup qualification":     3.0,
    "FIFA Series":                      4.0,
    "CONCACAF Series":                  4.0,
    "African Cup of Nations":           4.0,
    "Friendly":                         2.0,
    "Unity Cup":                        2.0,
    "Baltic Cup":                       2.0,
    "Diamond Jubilee International Football Tournament": 2.0,
    "Mukuru 4 Nations":                 2.0,
    "Tri-Nations Cup":                  2.0,
    "Morocco, Capital of African Football": 2.0,
}
ELO_K_BASE = 20.0  # base K, scaled by tourn_weight

# WC 2026 host nations get true_home_advantage when playing at home
WC2026_HOSTS = {"United States", "Mexico", "Canada"}

# confederation map (from Gulati, approximate)
CONFED_MAP = {}  # filled from Gulati dataset at load time


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def elo_update(ra: float, rb: float, score_a: float, score_b: float,
               k: float) -> tuple[float, float]:
    """Returns updated (ra, rb). score_a/b are goals (we derive result)."""
    if score_a > score_b:
        s_a, s_b = 1.0, 0.0
    elif score_a == score_b:
        s_a = s_b = 0.5
    else:
        s_a, s_b = 0.0, 1.0
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea
    return ra + k * (s_a - ea), rb + k * (s_b - eb)


class RollingWindow:
    """Maintains a sliding window of match outcomes for one team."""
    def __init__(self, initial_matches: list[dict]):
        # Each entry: {gf, ga, win, draw, loss, cs (clean sheet), btts}
        self.matches = list(initial_matches)

    def append(self, gf: int, ga: int):
        win  = 1 if gf > ga else 0
        draw = 1 if gf == ga else 0
        loss = 1 if gf < ga else 0
        cs   = 1 if ga == 0 else 0
        btts = 1 if gf > 0 and ga > 0 else 0
        self.matches.append({"gf": gf, "ga": ga, "win": win, "draw": draw,
                              "loss": loss, "cs": cs, "btts": btts})

    def stats(self, n: int) -> dict:
        """Compute rolling stats over the last n matches."""
        window = self.matches[-n:] if len(self.matches) >= n else self.matches
        m = len(window)
        if m == 0:
            return {
                "matches": 0, "win_rate": 0.5, "draw_rate": 0.2, "loss_rate": 0.3,
                "gf_avg": 1.2, "ga_avg": 1.2, "gd_avg": 0.0,
                "cs_rate": 0.2, "btts_rate": 0.4, "win_streak": 0, "scoring_rate": 0.7,
            }
        wins    = sum(x["win"]  for x in window)
        draws   = sum(x["draw"] for x in window)
        losses  = sum(x["loss"] for x in window)
        gf_tot  = sum(x["gf"]   for x in window)
        ga_tot  = sum(x["ga"]   for x in window)
        cs_tot  = sum(x["cs"]   for x in window)
        btts_t  = sum(x["btts"] for x in window)
        # Win streak: count consecutive wins from end
        streak = 0
        for x in reversed(window):
            if x["win"]: streak += 1
            else: break
        # scoring rate = fraction of matches where team scored >= 1
        sc = sum(1 for x in window if x["gf"] > 0)
        return {
            "matches":      m,
            "win_rate":     wins / m,
            "draw_rate":    draws / m,
            "loss_rate":    losses / m,
            "gf_avg":       gf_tot / m,
            "ga_avg":       ga_tot / m,
            "gd_avg":       (gf_tot - ga_tot) / m,
            "cs_rate":      cs_tot / m,
            "btts_rate":    btts_t / m,
            "win_streak":   streak,
            "scoring_rate": sc / m,
        }


class H2HRecord:
    """Head-to-head record between two teams (direction-aware)."""
    def __init__(self, h_wins: int, a_wins: int, draws: int):
        self.h_wins = h_wins
        self.a_wins = a_wins
        self.draws  = draws

    @property
    def total(self):
        return self.h_wins + self.a_wins + self.draws

    def stats(self):
        t = max(self.total, 1)
        return {
            "h2h_matches":       self.total,
            "h2h_home_win_rate": self.h_wins / t,
            "h2h_away_win_rate": self.a_wins / t,
            "h2h_draw_rate":     self.draws  / t,
        }

    def update(self, gf: int, ga: int):
        if gf > ga:   self.h_wins += 1
        elif gf < ga: self.a_wins += 1
        else:         self.draws  += 1


def load_gulati_state(gulati_path: Path):
    """
    From the Gulati dataset, extract the last-known state for each team:
      - elo
      - last match date
      - last ~20 match results (for rolling windows)
      - total matches
      - confederation
      - H2H lookup
      - penalty / shootout / owngoal rates
    """
    print("Loading Gulati dataset ...")
    df = pd.read_csv(gulati_path, parse_dates=["date"])

    # Item 9: Compute is_knockout for historical Gulati matches
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

    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} rows, last date: {df['date'].max().date()}")

    # Confederation map
    for _, row in df.iterrows():
        CONFED_MAP.setdefault(row["home_team"], row["home_confed"])
        CONFED_MAP.setdefault(row["away_team"], row["away_confed"])

    # Per-team state
    team_elo:         dict[str, float]        = {}
    team_last_date:   dict[str, pd.Timestamp] = {}
    team_total:       dict[str, int]          = {}
    team_history:     dict[str, list]         = {}  # list of {gf,ga,...}
    team_penalty:     dict[str, list]         = {}  # list of booleans (scored pen?)
    team_owngoal:     dict[str, list]         = {}
    team_shootout:    dict[str, list]         = {}  # (played, won)
    h2h:              dict[tuple, H2HRecord]  = {}

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        hg = row["home_score"]
        ag = row["away_score"]

        # Update elo from Gulati (last elo_home / elo_away values)
        team_elo[ht] = float(row["elo_home"])
        team_elo[at] = float(row["elo_away"])

        # Update dates and totals
        team_last_date[ht] = row["date"]
        team_last_date[at] = row["date"]

        # We'll approximate total matches from the dataset column
        team_total[ht] = int(row["home_total_matches"])
        team_total[at] = int(row["away_total_matches"])

        # Append to history (for initializing rolling windows)
        if pd.notna(hg) and pd.notna(ag):
            hg, ag = int(hg), int(ag)
            entry_h = {"gf": hg, "ga": ag,
                       "win": int(hg>ag), "draw": int(hg==ag), "loss": int(hg<ag),
                       "cs": int(ag==0), "btts": int(hg>0 and ag>0)}
            entry_a = {"gf": ag, "ga": hg,
                       "win": int(ag>hg), "draw": int(ag==hg), "loss": int(ag<hg),
                       "cs": int(hg==0), "btts": int(hg>0 and ag>0)}
            team_history.setdefault(ht, []).append(entry_h)
            team_history.setdefault(at, []).append(entry_a)

        # H2H
        key = (ht, at)
        rkey = (at, ht)
        if key not in h2h and rkey not in h2h:
            h2h[key] = H2HRecord(0, 0, 0)
        # Accumulate from raw data is expensive - instead we'll use the
        # last h2h columns from Gulati which are already computed
        # We just need to store H2H as-seen in the last Gulati encounter
        if key in h2h and pd.notna(hg) and pd.notna(ag):
            h2h[key].update(hg, ag)

    # Keep only last 20 entries per team (enough for L20 window)
    for t in team_history:
        team_history[t] = team_history[t][-20:]

    # Build shootout rates from Gulati columns
    team_shootout_wins = {}
    team_penalty_rel   = {}
    team_owngoal_rate  = {}
    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        team_shootout_wins.setdefault(ht, row.get("home_shootout_win_rate", 0.5))
        team_shootout_wins.setdefault(at, row.get("away_shootout_win_rate", 0.5))
        team_penalty_rel.setdefault(ht,   row.get("home_penalty_reliance", 0.0))
        team_penalty_rel.setdefault(at,   row.get("away_penalty_reliance", 0.0))
        team_owngoal_rate.setdefault(ht,  row.get("home_owngoal_benefit_rate", 0.0))
        team_owngoal_rate.setdefault(at,  row.get("away_owngoal_benefit_rate", 0.0))

    # Wrap history in RollingWindow objects
    rolling: dict[str, RollingWindow] = {}
    for t, hist in team_history.items():
        rolling[t] = RollingWindow(hist)

    return {
        "elo":        team_elo,
        "last_date":  team_last_date,
        "total":      team_total,
        "rolling":    rolling,
        "h2h":        h2h,
        "shootout":   team_shootout_wins,
        "penalty":    team_penalty_rel,
        "owngoal":    team_owngoal_rate,
        "gulati_df":  df,
    }


def build_feature_row(
    date: pd.Timestamp,
    home: str,
    away: str,
    home_score,
    away_score,
    tournament: str,
    neutral: bool,
    state: dict,
) -> dict:
    """Build one 102-column feature row for a 2026 match."""
    elo_h = state["elo"].get(home, 1500.0)
    elo_a = state["elo"].get(away, 1500.0)
    elo_diff = elo_h - elo_a
    elo_exp  = elo_expected(elo_h, elo_a)

    last_h = state["last_date"].get(home)
    last_a = state["last_date"].get(away)
    days_h = (date - last_h).days if last_h is not None else 365
    days_a = (date - last_a).days if last_a is not None else 365

    tot_h = state["total"].get(home, 0)
    tot_a = state["total"].get(away, 0)

    rw_h = state["rolling"].get(home, RollingWindow([]))
    rw_a = state["rolling"].get(away, RollingWindow([]))

    s5h = rw_h.stats(5);  s5a = rw_a.stats(5)
    s10h = rw_h.stats(10); s10a = rw_a.stats(10)
    s20h = rw_h.stats(20); s20a = rw_a.stats(20)

    # H2H — find matching record regardless of home/away direction
    h2h_key   = (home, away)
    h2h_rkey  = (away, home)
    if h2h_key in state["h2h"]:
        h2h_rec = state["h2h"][h2h_key]
        h2h_stats = h2h_rec.stats()
    elif h2h_rkey in state["h2h"]:
        rec = state["h2h"][h2h_rkey]
        t = max(rec.total, 1)
        h2h_stats = {
            "h2h_matches":       rec.total,
            "h2h_home_win_rate": rec.a_wins / t,
            "h2h_away_win_rate": rec.h_wins / t,
            "h2h_draw_rate":     rec.draws  / t,
        }
    else:
        h2h_stats = {"h2h_matches": 0, "h2h_home_win_rate": 0.33,
                     "h2h_away_win_rate": 0.33, "h2h_draw_rate": 0.34}

    twt = TOURN_WEIGHTS.get(tournament, 2.0)
    is_wc = 1 if "FIFA World Cup" in tournament and "qualification" not in tournament else 0
    is_qual = 1 if "qualification" in tournament else 0
    is_friendly = 1 if not is_wc and not is_qual else 0

    hc = CONFED_MAP.get(home, "UEFA")
    ac = CONFED_MAP.get(away, "UEFA")
    same_confed = 1 if hc == ac else 0

    true_home_adv = 0
    if not neutral and is_wc and home in WC2026_HOSTS:
        true_home_adv = 1

    is_ko = 0
    if is_wc == 1:
        y = date.year
        if y == 2014 and date >= pd.Timestamp("2014-06-28"):
            is_ko = 1
        elif y == 2018 and date >= pd.Timestamp("2018-06-30"):
            is_ko = 1
        elif y == 2022 and date >= pd.Timestamp("2022-12-03"):
            is_ko = 1
        elif y == 2026 and date >= pd.Timestamp("2026-07-04"):
            is_ko = 1

    result = np.nan
    if pd.notna(home_score) and pd.notna(away_score):
        hs, as_ = int(home_score), int(away_score)
        result = "H" if hs > as_ else ("D" if hs == as_ else "A")

    row = {
        "date": date.strftime("%Y-%m-%d"),
        "home_team": home,
        "away_team": away,
        "neutral": int(neutral),
        "tournament": tournament,
        "tourn_weight": twt,
        "is_world_cup": is_wc,
        "is_qualifier": is_qual,
        "is_friendly": is_friendly,
        "same_confederation": same_confed,
        "home_confed": hc,
        "away_confed": ac,
        "elo_home": round(elo_h, 4),
        "elo_away": round(elo_a, 4),
        "elo_diff": round(elo_diff, 4),
        "elo_expected_home": round(elo_exp, 6),
        "home_days_since_last": days_h,
        "away_days_since_last": days_a,
        "days_since_last_diff": days_h - days_a,
        "home_total_matches": tot_h,
        "away_total_matches": tot_a,
        "experience_diff": tot_h - tot_a,
        # L5
        "home_matches_L5":        s5h["matches"],
        "home_win_rate_L5":       s5h["win_rate"],
        "home_draw_rate_L5":      s5h["draw_rate"],
        "home_loss_rate_L5":      s5h["loss_rate"],
        "home_gf_avg_L5":         s5h["gf_avg"],
        "home_ga_avg_L5":         s5h["ga_avg"],
        "home_gd_avg_L5":         s5h["gd_avg"],
        "home_clean_sheet_rate_L5": s5h["cs_rate"],
        "home_btts_rate_L5":      s5h["btts_rate"],
        "home_win_streak_L5":     s5h["win_streak"],
        "home_scoring_rate_L5":   s5h["scoring_rate"],
        "away_matches_L5":        s5a["matches"],
        "away_win_rate_L5":       s5a["win_rate"],
        "away_draw_rate_L5":      s5a["draw_rate"],
        "away_loss_rate_L5":      s5a["loss_rate"],
        "away_gf_avg_L5":         s5a["gf_avg"],
        "away_ga_avg_L5":         s5a["ga_avg"],
        "away_gd_avg_L5":         s5a["gd_avg"],
        "away_clean_sheet_rate_L5": s5a["cs_rate"],
        "away_btts_rate_L5":      s5a["btts_rate"],
        "away_win_streak_L5":     s5a["win_streak"],
        "away_scoring_rate_L5":   s5a["scoring_rate"],
        "win_rate_diff_L5":       s5h["win_rate"] - s5a["win_rate"],
        "gd_avg_diff_L5":         s5h["gd_avg"]   - s5a["gd_avg"],
        "gf_avg_diff_L5":         s5h["gf_avg"]   - s5a["gf_avg"],
        # L10
        "home_matches_L10":        s10h["matches"],
        "home_win_rate_L10":       s10h["win_rate"],
        "home_draw_rate_L10":      s10h["draw_rate"],
        "home_loss_rate_L10":      s10h["loss_rate"],
        "home_gf_avg_L10":         s10h["gf_avg"],
        "home_ga_avg_L10":         s10h["ga_avg"],
        "home_gd_avg_L10":         s10h["gd_avg"],
        "home_clean_sheet_rate_L10": s10h["cs_rate"],
        "home_btts_rate_L10":      s10h["btts_rate"],
        "home_win_streak_L10":     s10h["win_streak"],
        "home_scoring_rate_L10":   s10h["scoring_rate"],
        "away_matches_L10":        s10a["matches"],
        "away_win_rate_L10":       s10a["win_rate"],
        "away_draw_rate_L10":      s10a["draw_rate"],
        "away_loss_rate_L10":      s10a["loss_rate"],
        "away_gf_avg_L10":         s10a["gf_avg"],
        "away_ga_avg_L10":         s10a["ga_avg"],
        "away_gd_avg_L10":         s10a["gd_avg"],
        "away_clean_sheet_rate_L10": s10a["cs_rate"],
        "away_btts_rate_L10":      s10a["btts_rate"],
        "away_win_streak_L10":     s10a["win_streak"],
        "away_scoring_rate_L10":   s10a["scoring_rate"],
        "win_rate_diff_L10":       s10h["win_rate"] - s10a["win_rate"],
        "gd_avg_diff_L10":         s10h["gd_avg"]   - s10a["gd_avg"],
        "gf_avg_diff_L10":         s10h["gf_avg"]   - s10a["gf_avg"],
        # L20
        "home_matches_L20":        s20h["matches"],
        "home_win_rate_L20":       s20h["win_rate"],
        "home_draw_rate_L20":      s20h["draw_rate"],
        "home_loss_rate_L20":      s20h["loss_rate"],
        "home_gf_avg_L20":         s20h["gf_avg"],
        "home_ga_avg_L20":         s20h["ga_avg"],
        "home_gd_avg_L20":         s20h["gd_avg"],
        "home_clean_sheet_rate_L20": s20h["cs_rate"],
        "home_btts_rate_L20":      s20h["btts_rate"],
        "home_win_streak_L20":     s20h["win_streak"],
        "home_scoring_rate_L20":   s20h["scoring_rate"],
        "away_matches_L20":        s20a["matches"],
        "away_win_rate_L20":       s20a["win_rate"],
        "away_draw_rate_L20":      s20a["draw_rate"],
        "away_loss_rate_L20":      s20a["loss_rate"],
        "away_gf_avg_L20":         s20a["gf_avg"],
        "away_ga_avg_L20":         s20a["ga_avg"],
        "away_gd_avg_L20":         s20a["gd_avg"],
        "away_clean_sheet_rate_L20": s20a["cs_rate"],
        "away_btts_rate_L20":      s20a["btts_rate"],
        "away_win_streak_L20":     s20a["win_streak"],
        "away_scoring_rate_L20":   s20a["scoring_rate"],
        "win_rate_diff_L20":       s20h["win_rate"] - s20a["win_rate"],
        "gd_avg_diff_L20":         s20h["gd_avg"]   - s20a["gd_avg"],
        "gf_avg_diff_L20":         s20h["gf_avg"]   - s20a["gf_avg"],
        # H2H
        **h2h_stats,
        # Penalty / shootout (carry-forward from Gulati)
        "home_penalty_reliance":       state["penalty"].get(home, 0.0),
        "away_penalty_reliance":       state["penalty"].get(away, 0.0),
        "home_owngoal_benefit_rate":   state["owngoal"].get(home, 0.0),
        "away_owngoal_benefit_rate":   state["owngoal"].get(away, 0.0),
        "home_shootout_win_rate":      state["shootout"].get(home, 0.5),
        "away_shootout_win_rate":      state["shootout"].get(away, 0.5),
        "true_home_advantage":  true_home_adv,
        "is_knockout": is_ko,
        # Labels (NaN for upcoming matches)
        "home_score": home_score,
        "away_score": away_score,
        "result": result,
    }
    return row


def update_state_after_match(
    state: dict,
    date: pd.Timestamp,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    tournament: str,
):
    """Mutate state to reflect a completed match result."""
    twt = TOURN_WEIGHTS.get(tournament, 2.0)
    k   = ELO_K_BASE * (twt / 2.0)

    elo_h = state["elo"].get(home, 1500.0)
    elo_a = state["elo"].get(away, 1500.0)
    new_h, new_a = elo_update(elo_h, elo_a, home_score, away_score, k)

    state["elo"][home]       = new_h
    state["elo"][away]       = new_a
    state["last_date"][home] = date
    state["last_date"][away] = date
    state["total"][home]     = state["total"].get(home, 0) + 1
    state["total"][away]     = state["total"].get(away, 0) + 1

    state["rolling"].setdefault(home, RollingWindow([])).append(home_score, away_score)
    state["rolling"].setdefault(away, RollingWindow([])).append(away_score, home_score)

    # H2H update
    key = (home, away)
    if key not in state["h2h"]:
        rkey = (away, home)
        if rkey in state["h2h"]:
            state["h2h"][rkey].update(away_score, home_score)
        else:
            state["h2h"][key] = H2HRecord(0, 0, 0)
            state["h2h"][key].update(home_score, away_score)
    else:
        state["h2h"][key].update(home_score, away_score)


def main():
    matches_path = Path("data/raw/matches_2026.csv")
    gulati_path  = config.GULATI_CSV
    out_path     = Path("data/processed/dataset_with_2026.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load state from Gulati ─────────────────────────────────────────────────
    state = load_gulati_state(gulati_path)
    gulati_df = state.pop("gulati_df")

    # ── Load 2026 matches ──────────────────────────────────────────────────────
    matches_2026 = pd.read_csv(matches_path)
    matches_2026["date"] = pd.to_datetime(matches_2026["date"])
    matches_2026 = matches_2026.sort_values("date").reset_index(drop=True)
    print(f"\nLoaded {len(matches_2026)} 2026 matches to process.")

    # ── Process each match chronologically ─────────────────────────────────────
    new_rows = []
    upcoming = []
    for _, m in matches_2026.iterrows():
        home = m["home_team"]
        away = m["away_team"]
        hs_raw = m["home_score"]
        as_raw = m["away_score"]
        tournament = m["tournament"]
        neutral = str(m.get("neutral", "TRUE")).upper() == "TRUE"
        date = m["date"]

        has_result = (
            pd.notna(hs_raw)
            and pd.notna(as_raw)
            and str(hs_raw).strip() not in ("", "NA", "nan")
            and str(as_raw).strip() not in ("", "NA", "nan")
        )

        # Build feature row BEFORE updating state (features reflect state
        # at kick-off, not after the match)
        hs = int(float(hs_raw)) if has_result else np.nan
        as_ = int(float(as_raw)) if has_result else np.nan

        row = build_feature_row(
            date=date, home=home, away=away,
            home_score=hs, away_score=as_,
            tournament=tournament, neutral=neutral,
            state=state,
        )
        new_rows.append(row)

        # After building the row, update state for future matches
        if has_result:
            update_state_after_match(
                state, date, home, away, int(float(hs_raw)), int(float(as_raw)), tournament
            )
        else:
            upcoming.append(f"  {date.date()}  {home} vs {away}  [{tournament}]")

    print(f"\nBuilt {len(new_rows)} new rows ({len(upcoming)} upcoming/unplayed).")
    if upcoming:
        print("Upcoming matches (to be predicted):")
        for u in upcoming:
            print(u)

    # ── Assemble final dataset ─────────────────────────────────────────────────
    new_df = pd.DataFrame(new_rows)

    # Align columns to Gulati schema
    gulati_cols = list(gulati_df.columns)
    for col in gulati_cols:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[gulati_cols]

    combined = pd.concat([gulati_df, new_df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined dataset -> {out_path}")
    print(f"  Total rows: {len(combined)}")
    print(f"  Gulati rows: {len(gulati_df)}")
    print(f"  New 2026 rows: {len(new_df)}")

    # Sanity check: Elo for key teams
    print("\nFinal Elo ratings for key 2026 WC teams:")
    teams_to_check = ["Brazil", "France", "Argentina", "Spain", "Portugal",
                      "England", "Germany", "Morocco", "United States", "Mexico"]
    for t in teams_to_check:
        elo = state["elo"].get(t)
        if elo:
            print(f"  {t:<20} {elo:.1f}")


if __name__ == "__main__":
    main()
