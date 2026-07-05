import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Allow importing from root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

def get_latest_team_stats(df: pd.DataFrame, team: str) -> dict:
    """
    Find the most recent match for a team and return their stats,
    mapping away stats to home if their last match was away (and vice versa).
    """
    # Filter matches involving the team
    team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")
    if len(team_matches) == 0:
        raise ValueError(f"Team '{team}' not found in the dataset. Please check spelling.")
    
    last_match = team_matches.iloc[-1]
    is_home = last_match["home_team"] == team
    
    stats = {}
    prefix_src = "home" if is_home else "away"
    
    # We want to extract all stats and prefix them as "target"
    for col in df.columns:
        # Elo
        if col == f"elo_{prefix_src}":
            stats["elo"] = last_match[col]
        # General prefix mapping (e.g. home_win_rate_L5 -> win_rate_L5)
        elif col.startswith(f"{prefix_src}_"):
            base_name = col[len(prefix_src)+1:]
            stats[base_name] = last_match[col]
            
    return stats

def main():
    parser = argparse.ArgumentParser(description="Add upcoming 2026 World Cup fixtures to the dataset.")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--date", required=True, help="Match date (YYYY-MM-DD)")
    parser.add_argument("--neutral", type=int, default=1, help="Neutral venue (1 for Yes, 0 for No)")
    parser.add_argument("--stage", default="Round of 16", help="Tournament stage (e.g., Round of 16, Quarter-finals, Semi-finals, Final)")
    args = parser.parse_args()

    csv_path = config.GULATI_CSV
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    try:
        home_stats = get_latest_team_stats(df, args.home)
        away_stats = get_latest_team_stats(df, args.away)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Creating fixture: {args.home} vs {args.away} on {args.date} ({args.stage})...")

    # Construct the new row
    new_row = {}
    
    # Metadata columns
    new_row["date"] = pd.to_datetime(args.date)
    new_row["home_team"] = args.home
    new_row["away_team"] = args.away
    new_row["home_score"] = np.nan
    new_row["away_score"] = np.nan
    new_row["result"] = np.nan
    new_row["tournament"] = "FIFA World Cup"
    new_row["is_world_cup"] = 1
    new_row["is_qualifier"] = 0
    new_row["is_friendly"] = 0
    
    # Extract confed from last matches
    home_matches = df[(df["home_team"] == args.home) | (df["away_team"] == args.home)]
    away_matches = df[(df["home_team"] == args.away) | (df["away_team"] == args.away)]
    
    h_last = home_matches.iloc[-1]
    a_last = away_matches.iloc[-1]
    
    home_conf = h_last["home_confed"] if h_last["home_team"] == args.home else h_last["away_confed"]
    away_conf = a_last["home_confed"] if a_last["home_team"] == args.away else a_last["away_confed"]
    
    new_row["home_confed"] = home_conf
    new_row["away_confed"] = away_conf
    new_row["same_confederation"] = 1 if home_conf == away_conf else 0
    new_row["neutral"] = args.neutral
    new_row["tourn_weight"] = 8.0  # WC final weight
    
    # Home advantage (host nation advantage)
    # WC 2026 hosts are USA, Mexico, Canada
    new_row["true_home_advantage"] = 0
    if args.neutral == 0 and args.home in ["United States", "Mexico", "Canada"]:
         new_row["true_home_advantage"] = 1

    # Populate home-prefixed features
    for k, v in home_stats.items():
        if k == "elo":
            new_row["elo_home"] = v
        else:
            new_row[f"home_{k}"] = v

    # Populate away-prefixed features
    for k, v in away_stats.items():
        if k == "elo":
            new_row["elo_away"] = v
        else:
            new_row[f"away_{k}"] = v

    # Calculate difference and expected outcome features
    new_row["elo_diff"] = new_row["elo_home"] - new_row["elo_away"]
    new_row["elo_expected_home"] = 1 / (10 ** (-new_row["elo_diff"] / 400) + 1)
    
    # Calculate days since last match
    # Assuming match is played on args.date
    last_h_date = pd.to_datetime(h_last["date"])
    last_a_date = pd.to_datetime(a_last["date"])
    new_row["home_days_since_last"] = (new_row["date"] - last_h_date).days
    new_row["away_days_since_last"] = (new_row["date"] - last_a_date).days
    new_row["days_since_last_diff"] = new_row["home_days_since_last"] - new_row["away_days_since_last"]
    
    # Experience and fatigue
    new_row["experience_diff"] = new_row["home_total_matches"] - new_row["away_total_matches"]
    
    # Form differences
    new_row["win_rate_diff_L5"] = new_row["home_win_rate_L5"] - new_row["away_win_rate_L5"]
    new_row["gd_avg_diff_L5"] = new_row["home_gd_avg_L5"] - new_row["away_gd_avg_L5"]
    new_row["gf_avg_diff_L5"] = new_row["home_gf_avg_L5"] - new_row["away_gf_avg_L5"]
    
    new_row["win_rate_diff_L10"] = new_row["home_win_rate_L10"] - new_row["away_win_rate_L10"]
    new_row["gd_avg_diff_L10"] = new_row["home_gd_avg_L10"] - new_row["away_gd_avg_L10"]
    new_row["gf_avg_diff_L10"] = new_row["home_gf_avg_L10"] - new_row["away_gf_avg_L10"]
    
    new_row["win_rate_diff_L20"] = new_row["home_win_rate_L20"] - new_row["away_win_rate_L20"]
    new_row["gd_avg_diff_L20"] = new_row["home_gd_avg_L20"] - new_row["away_gd_avg_L20"]
    new_row["gf_avg_diff_L20"] = new_row["home_gf_avg_L20"] - new_row["away_gf_avg_L20"]

    # Fill any remaining columns not populated (use default 0 or copy from home/away)
    new_row_df = pd.DataFrame([new_row])
    for col in df.columns:
        if col not in new_row_df.columns:
            new_row_df[col] = 0.0

    # Ensure column order matches exactly
    new_row_df = new_row_df[df.columns]

    # Append to CSV
    updated_df = pd.concat([df, new_row_df], ignore_index=True)
    # Sort by date
    updated_df = updated_df.sort_values("date").reset_index(drop=True)
    updated_df.to_csv(csv_path, index=False)

    print(f"Successfully added match and saved to {csv_path}")

if __name__ == "__main__":
    main()
