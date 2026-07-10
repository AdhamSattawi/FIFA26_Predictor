"""
scrape_lineups.py — Scrape starting XIs from Transfermarkt for every match
in the Gulati dataset that falls within our training/validation scope.

Scope:
  - World Cup 2014, 2018, 2022 (all matches)
  - WC Qualifiers for each cycle (all confederations)
  - International Friendlies within 12 months before each WC

Output: data/raw/lineups/all_lineups.csv
"""

import sys
import re
import time
import random
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.scraping.utils import (
    polite_sleep, new_browser_context,
    safe_inner_text, safe_get_attribute,
    normalize_team_name, dismiss_consent, log,
)

# ── Output ────────────────────────────────────────────────────────────────────
LINEUPS_CSV = config.LINEUPS_CSV

# ── WC dates (approximate tournament start dates) ─────────────────────────────
WC_DATES = {
    2014: datetime(2014, 6, 12),
    2018: datetime(2018, 6, 14),
    2022: datetime(2022, 11, 20),
    2026: datetime(2026, 6, 11),
}


def get_match_scope(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the Gulati dataset to only the matches we want lineups for.
    Additionally appends completed 2026 World Cup matches from matches_2026.csv.
    """
    df["date"] = pd.to_datetime(df["date"])
    mask_wc  = (df["is_world_cup"] == 1) & df["date"].dt.year.isin([2014, 2018, 2022])
    historical_wc = df[mask_wc].copy()
    historical_wc["is_qualifier"] = 0
    historical_wc["is_friendly"] = 0

    # Load 2026 matches
    m2026_path = Path("data/raw/matches_2026.csv")
    if m2026_path.exists():
        df26 = pd.read_csv(m2026_path, parse_dates=["date"])
        # Only completed World Cup matches
        completed_wc26 = df26[
            (df26["tournament"] == "FIFA World Cup") &
            df26["home_score"].notna() &
            (df26["home_score"].astype(str).str.strip() != "")
        ].copy()
        completed_wc26["is_world_cup"] = 1
        completed_wc26["is_qualifier"] = 0
        completed_wc26["is_friendly"] = 0
        
        # Merge them
        scope = pd.concat([historical_wc, completed_wc26], ignore_index=True)
    else:
        scope = historical_wc

    return scope.reset_index(drop=True)


def build_tm_search_url(home_team: str, away_team: str, date: str) -> str:
    """
    Build a Transfermarkt search URL for a given match.
    We search by date on the spielplan pages; lineup pages require a match ID.
    This function is a helper — the main flow finds match IDs from schedule pages.
    """
    return (
        f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
        f"?query={home_team}+vs+{away_team}&x=0&y=0"
    )


def scrape_lineup_from_page(page, match_id: str, home_team: str, away_team: str) -> list[dict]:
    """
    Given a Transfermarkt lineup page already loaded, extract starting XI for
    both teams. Returns a list of player dicts.
    """
    players = []
    try:
        tables = page.locator("table.items").all()
        if len(tables) < 2:
            log.warning(f"  Only {len(tables)} lineup table(s) found for match {match_id}")
            return players

        # Table 0 is home starting XI, Table 1 is away starting XI
        for idx, table in enumerate(tables[:2]):
            team_name = home_team if idx == 0 else away_team
            rows = table.locator("tbody > tr").all()
            
            # Each player takes 3 rows in the new layout
            for i in range(0, len(rows), 3):
                if i >= len(rows):
                    break
                row = rows[i]
                cells = row.locator("td").all()
                if len(cells) < 5:
                    continue
                
                shirt_number = safe_inner_text(cells[0])
                
                # Player name and ID from cell 3
                name_anchor = cells[3].locator("a").first
                player_name = safe_inner_text(name_anchor)
                player_href = safe_get_attribute(name_anchor, "href")
                
                # Position from cell 4
                position_raw = safe_inner_text(cells[4])
                position = position_raw.split(",")[0].strip()
                
                # Extract player_id
                pid_match = re.search(r"/spieler/(\d+)", player_href)
                player_id = pid_match.group(1) if pid_match else ""
                
                if player_name:
                    players.append({
                        "match_id":     match_id,
                        "team":         team_name,
                        "player_name":  player_name,
                        "player_id":    player_id,
                        "position":     position,
                        "shirt_number": shirt_number,
                    })
    except Exception as e:
        log.error(f"  Error parsing lineup for match {match_id}: {e}")

    return players


def build_schedule_cache(page, competition_id: str, saison: int) -> dict:
    """
    Load the schedule page once and parse all matches to build a lookup cache:
    {(home_team_normalized, away_team_normalized, date_str): match_id}
    """
    cache = {}
    if competition_id in ("WM14", "WM18", "WM22", "WM26"):
        url = f"https://www.transfermarkt.com/world-cup/gesamtspielplan/pokalwettbewerb/FIWC/saison_id/{saison}"
    else:
        url = (
            f"https://www.transfermarkt.com/x/spielplan/wettbewerb/{competition_id}"
            f"/plus/?saison_id={saison}"
        )

    log.info(f"  Building schedule cache from: {url}")
    try:
        page.goto(url, timeout=30000)
        dismiss_consent(page)
        page.wait_for_selector("table.spielplan-ergebnis, div.box, table.items, tr", timeout=15000)
    except Exception as e:
        log.warning(f"  Could not load schedule page {url}: {e}")
        return cache

    # Find all match rows
    if competition_id in ("WM14", "WM18", "WM22"):
        rows = page.locator("tr").filter(has=page.locator("a[href*='spielbericht']")).all()
    else:
        rows = page.locator("table.spielplan-ergebnis tbody tr").all()

    log.info(f"  Found {len(rows)} match rows on the schedule page.")
    for row in rows:
        try:
            if competition_id in ("WM14", "WM18", "WM22"):
                row_date_str = safe_inner_text(row.locator("td").nth(0))
            else:
                row_date_str = safe_inner_text(row.locator("td").nth(1))
                
            row_home = normalize_team_name(
                safe_get_attribute(row.locator("a[title]").first, "title")
            )
            row_away = normalize_team_name(
                safe_get_attribute(row.locator("a[title]").last, "title")
            )
            # Parse date
            try:
                row_date = pd.to_datetime(row_date_str, dayfirst=True)
            except Exception:
                continue

            # Get match report link
            match_link = row.locator("a[href*='spielbericht']").first
            href = safe_get_attribute(match_link, "href")
            mid_match = re.search(r"/spielbericht/(\d+)", href)
            if mid_match:
                match_id = mid_match.group(1)
                key = (row_home.lower(), row_away.lower(), row_date.strftime("%Y-%m-%d"))
                cache[key] = match_id
        except Exception:
            continue

    return cache


def lookup_match_id(cache: dict, home_team: str, away_team: str, date: pd.Timestamp) -> str | None:
    """Find a match ID from cache with flexible team names and +/- 1 day date tolerance."""
    home_norm = normalize_team_name(home_team).lower()
    away_norm = normalize_team_name(away_team).lower()
    
    for (cached_home, cached_away, cached_date_str), match_id in cache.items():
        try:
            cached_date = pd.to_datetime(cached_date_str)
        except Exception:
            continue
            
        date_match = abs((cached_date - date).days) <= 1
        teams_match = (
            (cached_home in home_norm or home_norm in cached_home)
            and
            (cached_away in away_norm or away_norm in cached_away)
        )
        if date_match and teams_match:
            return match_id
    return None


def run_scraper(competition_id: str, matches_df: pd.DataFrame, page) -> list[dict]:
    """
    For each match in matches_df, lookup its Transfermarkt match ID using a cached schedule,
    then scrape the lineup page. Returns list of player dicts.
    """
    all_players = []
    if len(matches_df) == 0:
        return []

    # Get season ID from first match in df
    first_date = pd.to_datetime(matches_df.iloc[0]["date"])
    saison = first_date.year - 1 if first_date.month < 7 else first_date.year

    # Load and cache matches schedule
    cache = build_schedule_cache(page, competition_id, saison)
    if not cache:
        log.warning(f"  Schedule cache empty for {competition_id} saison {saison}. Skipping all matches.")
        return []

    for _, row in matches_df.iterrows():
        home  = str(row["home_team"])
        away  = str(row["away_team"])
        date  = pd.to_datetime(row["date"])

        log.info(f"  [{competition_id}] {date.date()} {home} vs {away}")

        match_id = lookup_match_id(cache, home, away, date)
        if not match_id:
            log.warning(f"    → Match ID not found in cache, skipping.")
            continue

        lineup_url = (
            f"https://www.transfermarkt.com/x/aufstellung/spielbericht/{match_id}"
        )
        try:
            page.goto(lineup_url, timeout=30000)
            dismiss_consent(page)
            page.wait_for_selector("table.items", timeout=10000)
        except (PWTimeout, Exception) as e:
            log.warning(f"    → Lineup page load failed ({e}), skipping.")
            polite_sleep(2.0, 4.0)
            continue

        players = scrape_lineup_from_page(page, match_id, home, away)
        # Attach match metadata
        for p in players:
            p["match_date"]  = date.strftime("%Y-%m-%d")
            p["home_team"]   = home
            p["away_team"]   = away
            p["competition"] = competition_id

        all_players.extend(players)
        log.info(f"    → {len(players)} player records scraped.")
        polite_sleep()

    return all_players


def main():
    log.info("Loading Gulati dataset ...")
    df = pd.read_csv(config.GULATI_CSV)
    scope_df = get_match_scope(df)
    log.info(f"  → Total scope matches for lineup scraping: {len(scope_df)}")

    # Check for already scraped matches to enable incremental scraping
    existing_keys = set()
    if LINEUPS_CSV.exists():
        try:
            existing_df = pd.read_csv(LINEUPS_CSV)
            existing_df["home_norm"] = existing_df["home_team"].apply(normalize_team_name).str.lower()
            existing_df["away_norm"] = existing_df["away_team"].apply(normalize_team_name).str.lower()
            existing_df["date_str"] = pd.to_datetime(existing_df["match_date"]).dt.strftime("%Y-%m-%d")
            existing_keys = set(zip(existing_df["home_norm"], existing_df["away_norm"], existing_df["date_str"]))
            log.info(f"  → Loaded {len(existing_keys)} already scraped matches from {LINEUPS_CSV}.")
        except Exception as e:
            log.warning(f"  → Could not load existing lineups file: {e}. Scraping all matches.")

    # Filter to only matches that haven\'t been scraped
    scope_df["home_norm"] = scope_df["home_team"].apply(normalize_team_name).str.lower()
    scope_df["away_norm"] = scope_df["away_team"].apply(normalize_team_name).str.lower()
    scope_df["date_str"] = scope_df["date"].dt.strftime("%Y-%m-%d")

    def is_new(row):
        key = (row["home_norm"], row["away_norm"], row["date_str"])
        return key not in existing_keys

    new_scope_df = scope_df[scope_df.apply(is_new, axis=1)].copy().reset_index(drop=True)
    log.info(f"  → {len(new_scope_df)} matches need to be scraped (out of {len(scope_df)} total).")

    if len(new_scope_df) == 0:
        log.info("✓ All matches in scope are already scraped. Nothing to do.")
        return

    # Group by competition type for targeted scraping
    wc_df    = new_scope_df[new_scope_df["is_world_cup"] == 1]
    qual_df  = new_scope_df[new_scope_df["is_qualifier"] == 1]
    friendly_df = new_scope_df[new_scope_df["is_friendly"] == 1]

    config.LINEUPS_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_players: list[dict] = []

    with sync_playwright() as p:
        browser, context, page = new_browser_context(p)

        # Scrape World Cup matches
        log.info(f"\n=== World Cup Matches ({len(wc_df)}) ===")
        for wc_year, wc_info in config.WC_COMPETITIONS.items():
            wc_mask = wc_df["date"].dt.year.between(
                wc_info["year"], wc_info["year"]
            )
            wc_subset = wc_df[wc_mask]
            if len(wc_subset):
                all_players.extend(
                    run_scraper(wc_info["id"], wc_subset, page)
                )

        # Scrape Qualifier matches
        log.info(f"\n=== Qualifier Matches ({len(qual_df)}) ===")
        for conf, qid in config.QUALIFIER_IDS.items():
            conf_mask = (qual_df["home_confed"] == conf) | (qual_df["away_confed"] == conf)
            q_subset = qual_df[conf_mask]
            if len(q_subset):
                log.info(f"  {conf}: {len(q_subset)} matches")
                all_players.extend(run_scraper(qid, q_subset, page))

        # Scrape Friendly matches
        log.info(f"\n=== Friendly Matches ({len(friendly_df)}) ===")
        all_players.extend(
            run_scraper(config.FRIENDLY_ID, friendly_df, page)
        )

        browser.close()

    if all_players:
        out_df = pd.DataFrame(all_players)
        if LINEUPS_CSV.exists():
            out_df.to_csv(LINEUPS_CSV, mode="a", header=False, index=False)
        else:
            out_df.to_csv(LINEUPS_CSV, index=False)
        log.info(f"\n✓ Saved {len(out_df)} new player-lineup records to {LINEUPS_CSV}")
    else:
        log.warning("No players scraped. Check Transfermarkt access.")


if __name__ == "__main__":
    main()
