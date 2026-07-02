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
    normalize_team_name, log,
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
    Filter the Gulati dataset to only the matches we want lineups for:
      - All WC matches (is_world_cup == 1)
      - All qualifier matches (is_qualifier == 1)
      - Friendlies within 12 months before each WC
    """
    df["date"] = pd.to_datetime(df["date"])
    mask_wc  = df["is_world_cup"] == 1
    mask_qual = df["is_qualifier"] == 1

    # Friendlies within FRIENDLY_WINDOW_MONTHS before each WC
    friendly_masks = []
    for wc_year, wc_date in WC_DATES.items():
        window_start = wc_date - timedelta(days=config.FRIENDLY_WINDOW_MONTHS * 30)
        m = (
            (df["is_friendly"] == 1)
            & (df["date"] >= window_start)
            & (df["date"] <= wc_date)
        )
        friendly_masks.append(m)

    import functools, operator
    mask_friendly = functools.reduce(operator.or_, friendly_masks)

    return df[mask_wc | mask_qual | mask_friendly].copy().reset_index(drop=True)


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


def scrape_lineup_from_page(page, match_id: str) -> list[dict]:
    """
    Given a Transfermarkt lineup page already loaded, extract starting XI for
    both teams. Returns a list of player dicts.
    """
    players = []
    try:
        # Both team lineup boxes
        lineup_boxes = page.locator("div.aufstellung-vereinssektion").all()
        if len(lineup_boxes) < 2:
            log.warning(f"  Only {len(lineup_boxes)} lineup box(es) found for match {match_id}")

        for box_idx, box in enumerate(lineup_boxes[:2]):
            team_name_raw = safe_inner_text(box.locator("a.vereinprofil_tooltip").first)
            team_name = normalize_team_name(team_name_raw)

            player_rows = box.locator("table.items tbody tr").all()
            for row in player_rows:
                # Each row has: shirt number | position | player name
                cells = row.locator("td").all()
                if len(cells) < 3:
                    continue
                shirt_number = safe_inner_text(cells[0])
                position     = safe_inner_text(cells[1])
                name_cell    = cells[2]
                player_name  = safe_inner_text(name_cell.locator("a").first)
                player_href  = safe_get_attribute(name_cell.locator("a").first, "href")

                # Extract player_id from href like /player-name/profil/spieler/12345
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


def find_match_id_on_tm(page, home_team: str, away_team: str,
                         date: pd.Timestamp, competition_id: str) -> str | None:
    """
    Try to find the Transfermarkt match ID for a given match by browsing
    the competition schedule page for that date's matchday.

    Returns the match_id string or None if not found.
    """
    # Transfermarkt spielplan URL for the competition
    year = date.year
    # For WC: e.g. WM22 saison 2022; for qualifiers: the relevant saison year
    saison = year - 1 if date.month < 7 else year
    url = (
        f"https://www.transfermarkt.com/x/spielplan/wettbewerb/{competition_id}"
        f"/plus/?saison_id={saison}"
    )

    try:
        page.goto(url, timeout=30000)
        page.wait_for_selector("table.spielplan-ergebnis", timeout=10000)
    except (PWTimeout, Exception) as e:
        log.warning(f"  Could not load schedule page {url}: {e}")
        return None

    # Find all match rows and match by teams + date
    rows = page.locator("table.spielplan-ergebnis tbody tr").all()
    for row in rows:
        try:
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

            date_match  = abs((row_date - date).days) <= 1
            teams_match = (
                (row_home.lower() in home_team.lower() or home_team.lower() in row_home.lower())
                and
                (row_away.lower() in away_team.lower() or away_team.lower() in row_away.lower())
            )

            if date_match and teams_match:
                # Get match report link
                match_link = row.locator("a[href*='spielbericht']").first
                href = safe_get_attribute(match_link, "href")
                mid_match = re.search(r"/spielbericht/(\d+)", href)
                if mid_match:
                    return mid_match.group(1)
        except Exception:
            continue

    return None


def run_scraper(competition_id: str, matches_df: pd.DataFrame, page) -> list[dict]:
    """
    For each match in matches_df, find its Transfermarkt match ID, then
    scrape the lineup page. Returns list of player dicts.
    """
    all_players = []

    for _, row in matches_df.iterrows():
        home  = str(row["home_team"])
        away  = str(row["away_team"])
        date  = pd.to_datetime(row["date"])

        log.info(f"  [{competition_id}] {date.date()} {home} vs {away}")

        match_id = find_match_id_on_tm(page, home, away, date, competition_id)
        if not match_id:
            log.warning(f"    → Match ID not found, skipping.")
            polite_sleep(1.0, 2.0)
            continue

        lineup_url = (
            f"https://www.transfermarkt.com/x/aufstellung/spielbericht/{match_id}"
        )
        try:
            page.goto(lineup_url, timeout=30000)
            page.wait_for_selector("div.aufstellung-vereinssektion", timeout=10000)
        except (PWTimeout, Exception) as e:
            log.warning(f"    → Lineup page load failed ({e}), skipping.")
            polite_sleep(2.0, 4.0)
            continue

        players = scrape_lineup_from_page(page, match_id)
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
    log.info("Loading Gulati dataset …")
    df = pd.read_csv(config.GULATI_CSV)
    scope_df = get_match_scope(df)
    log.info(f"  → {len(scope_df)} matches in scope for lineup scraping.")

    # Group by competition type for targeted scraping
    wc_df    = scope_df[scope_df["is_world_cup"] == 1]
    qual_df  = scope_df[scope_df["is_qualifier"] == 1]
    friendly_df = scope_df[scope_df["is_friendly"] == 1]

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
        out_df.to_csv(LINEUPS_CSV, index=False)
        log.info(f"\n✓ Saved {len(out_df)} player-lineup records to {LINEUPS_CSV}")
    else:
        log.warning("No players scraped. Check Transfermarkt access.")


if __name__ == "__main__":
    main()
