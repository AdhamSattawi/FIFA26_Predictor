"""
scrape_lineups_full.py — FULL COVERAGE lineup scraper.

Scrapes starting XIs from Transfermarkt for ALL matches in the Gulati dataset
training/validation scope, not just World Cup final matches:

  Scope per WC cycle (2014, 2018, 2022):
  ├─ World Cup final tournament (64 matches × 3 = 192)
  ├─ WC Qualifiers per confederation (AFC/CAF/CONCACAF/CONMEBOL/OFC/UEFA × 3 cycles)
  └─ International Friendlies within 12 months before each WC

Total estimated: ~5,900 matches (~5.5–7 hours at polite request rate).

CRITICAL DESIGN DECISIONS
--------------------------
1. RESUME SAFE: The script tracks completed matches in a progress file
   (data/raw/lineups/scrape_progress.csv). If interrupted, re-running
   continues from where it left off. Match IDs are logged as soon as found.

2. INCREMENTAL SAVES: Player records are appended to all_lineups.csv every
   50 matches (not just at the end) to minimize data loss on crash.

3. SMART SCOPE: "Other" confederation qualifiers (AFCON qual, Euro qual, etc.)
   are SKIPPED — they're not on Transfermarkt WC competition pages and are
   irrelevant to WC prediction.

4. CONFEDERATION MAPPING: Each confederation maps to a specific Transfermarkt
   competition ID per WC cycle. We build one schedule cache per competition
   (not per match) to avoid redundant page loads.

5. FRIENDLY SCOPE: Only the 12 months before each WC to avoid scraping
   thousands of old friendlies.

Usage:
  python src/scraping/scrape_lineups_full.py

Output:
  data/raw/lineups/all_lineups.csv        (all player records)
  data/raw/lineups/scrape_progress.csv    (match-level progress log)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.scraping.utils import (
    polite_sleep, new_browser_context,
    safe_inner_text, safe_get_attribute,
    normalize_team_name, dismiss_consent, log, USER_AGENT,
)

# ── Output paths ───────────────────────────────────────────────────────────────
LINEUPS_CSV  = config.LINEUPS_CSV                                   # all player records
PROGRESS_CSV = config.LINEUPS_DIR / "scrape_progress.csv"          # match-level progress

# ── WC tournament start dates ─────────────────────────────────────────────────
WC_DATES = {
    2014: datetime(2014, 6, 12),
    2018: datetime(2018, 6, 14),
    2022: datetime(2022, 11, 20),
}

# ── Transfermarkt competition IDs per WC cycle ───────────────────────────────
# World Cup (final tournament)
WC_COMP = {
    2014: {"id": "FIWC", "saison": 2013, "url_name": "world-cup"},
    2018: {"id": "FIWC", "saison": 2017, "url_name": "world-cup"},
    2022: {"id": "FIWC", "saison": 2022, "url_name": "world-cup"},
}

# WC Qualifiers: confederation → {wc_year → (competition_id, saison)}
# Sources verified against transfermarkt.com competition pages
QUALIFIER_COMPS = {
    "UEFA": {
        2014: ("WMQE", 2012),
        2018: ("WMQE", 2016),
        2022: ("WMQE", 2020),
    },
    "CONMEBOL": {
        2014: ("SA1Q", 2011),
        2018: ("SA1Q", 2015),
        2022: ("SA1Q", 2019),
    },
    "CONCACAF": {
        2014: ("CONC", 2012),
        2018: ("CONC", 2016),
        2022: ("CONC", 2019),
    },
    "AFC": {
        2014: ("AS1Q", 2011),
        2018: ("AS1Q", 2015),
        2022: ("AS1Q", 2019),
    },
    "CAF": {
        2014: ("AF1Q", 2011),
        2018: ("AF1Q", 2015),
        2022: ("AF1Q", 2019),
    },
    "OFC": {
        2014: ("OC1Q", 2011),
        2018: ("OC1Q", 2015),
        2022: ("OC1Q", 2019),
    },
    # Inter-confederation playoffs (different IDs per cycle)
    "Playoffs": {
        2014: ("WMQ2", 2013),
        2018: ("WMQ2", 2017),
        2022: ("WMQ2", 2021),
    },
}

# Friendly competition ID (international friendlies)
FRIENDLY_COMP_ID = "ISL"

# Checkpoint: save player records every N matches processed
SAVE_EVERY_N = 50

# Browser context refresh every N matches (prevents memory leaks)
REFRESH_EVERY_N = 60


# ── Scope filtering ────────────────────────────────────────────────────────────

def get_full_match_scope(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter Gulati dataset to all matches we want lineups for across 3 WC cycles.
    Assigns 'wc_cycle' (2014/2018/2022) and 'competition_group' to each match.
    Skips:
      - "Other" confederation qualifiers (AFCON qual, Euro qual, etc.)
      - Matches outside cycle windows
    """
    df["date"] = pd.to_datetime(df["date"])
    scope_rows = []

    for wc_year, wc_date in WC_DATES.items():
        prev_wc_date = WC_DATES.get(wc_year - 4, datetime(wc_year - 4, 1, 1))
        friendly_window_start = wc_date - timedelta(days=365)

        # ── World Cup final matches ────────────────────────────────────────────
        wc = df[(df["is_world_cup"] == 1) & (df["date"].dt.year == wc_year)].copy()
        wc["wc_cycle"] = wc_year
        wc["competition_group"] = "wc"
        scope_rows.append(wc)

        # ── Qualifiers (skip "Other" confederation — not on TM WC qual pages) ──
        qual = df[
            (df["is_qualifier"] == 1) &
            (df["date"] >= prev_wc_date) &
            (df["date"] < wc_date)
        ].copy()
        # Only include known confederations that map to TM WC qualifier pages
        known_confeds = set(QUALIFIER_COMPS.keys()) - {"Playoffs"}
        qual = qual[
            qual["home_confed"].isin(known_confeds) |
            qual["away_confed"].isin(known_confeds)
        ].copy()
        qual["wc_cycle"] = wc_year
        qual["competition_group"] = "qualifier"
        scope_rows.append(qual)

        # ── Friendlies (only within 12 months before WC) ──────────────────────
        fr = df[
            (df["is_friendly"] == 1) &
            (df["date"] >= friendly_window_start) &
            (df["date"] < wc_date)
        ].copy()
        fr["wc_cycle"] = wc_year
        fr["competition_group"] = "friendly"
        scope_rows.append(fr)

    scope = pd.concat(scope_rows, ignore_index=True)
    # Deduplicate (a match could appear in multiple cycles via date overlaps)
    scope = scope.drop_duplicates(subset=["date", "home_team", "away_team"]).copy()
    scope = scope.sort_values(["wc_cycle", "competition_group", "date"]).reset_index(drop=True)
    return scope


# ── Progress tracking ──────────────────────────────────────────────────────────

def load_progress() -> set:
    """Load set of already-processed match keys: '{date}|{home}|{away}'."""
    if not PROGRESS_CSV.exists():
        return set()
    try:
        prog = pd.read_csv(PROGRESS_CSV)
        return set(prog["match_key"].tolist())
    except Exception:
        return set()


def log_progress(match_key: str, match_id: str | None, n_players: int):
    """Append a single progress record to the progress CSV."""
    row = pd.DataFrame([{
        "match_key": match_key,
        "match_id": match_id or "",
        "n_players": n_players,
    }])
    row.to_csv(PROGRESS_CSV, mode="a", header=not PROGRESS_CSV.exists(), index=False)


def save_players(players: list[dict]):
    """Append player records to the lineups CSV."""
    if not players:
        return
    df = pd.DataFrame(players)
    df.to_csv(LINEUPS_CSV, mode="a", header=not LINEUPS_CSV.exists(), index=False)


# ── Schedule cache builders ────────────────────────────────────────────────────

def build_wc_schedule_cache(page, wc_year: int) -> dict:
    """
    Build match ID cache for a World Cup final tournament.
    Returns {(home_norm, away_norm, date_str): match_id}.
    """
    info = WC_COMP[wc_year]
    url = (
        f"https://www.transfermarkt.com/{info['url_name']}"
        f"/gesamtspielplan/pokalwettbewerb/{info['id']}"
        f"/saison_id/{info['saison']}"
    )
    return _build_cache_from_spielplan(page, url, is_wc=True)


def build_qualifier_schedule_cache(page, comp_id: str, saison: int) -> dict:
    """
    Build match ID cache for a WC qualifier competition.
    Returns {(home_norm, away_norm, date_str): match_id}.
    """
    url = (
        f"https://www.transfermarkt.com/x/spielplan/wettbewerb/{comp_id}"
        f"/plus/?saison_id={saison}"
    )
    return _build_cache_from_spielplan(page, url, is_wc=False)


def build_friendly_schedule_cache(page, saison: int) -> dict:
    """
    Build match ID cache for international friendlies for a given year.
    Returns {(home_norm, away_norm, date_str): match_id}.
    """
    url = (
        f"https://www.transfermarkt.com/x/spielplan/wettbewerb/{FRIENDLY_COMP_ID}"
        f"/plus/?saison_id={saison}"
    )
    return _build_cache_from_spielplan(page, url, is_wc=False)


def _build_cache_from_spielplan(page, url: str, is_wc: bool) -> dict:
    """
    Internal helper: load a schedule page and extract (team_pair, date) → match_id.
    Handles both WC-style spielplan and qualifier/friendly spielplan layouts.
    """
    cache = {}
    log.info(f"  Loading schedule cache: {url}")
    try:
        page.goto(url, timeout=45000)
        dismiss_consent(page)
        page.wait_for_selector("a[href*='spielbericht']", timeout=20000)
    except Exception as e:
        log.warning(f"  Schedule page failed: {e}")
        return cache

    # Collect all rows that have a spielbericht link
    try:
        rows = page.locator("tr").filter(
            has=page.locator("a[href*='spielbericht']")
        ).all()
    except Exception:
        rows = []

    log.info(f"  → {len(rows)} match rows found.")

    for row in rows:
        try:
            # Extract date
            date_cell = row.locator("td").first
            date_str_raw = safe_inner_text(date_cell)
            try:
                match_date = pd.to_datetime(date_str_raw, dayfirst=True)
            except Exception:
                continue

            # Extract teams from title attributes on anchor tags
            anchors_with_title = row.locator("a[title]").all()
            if len(anchors_with_title) < 2:
                continue
            home_title = safe_get_attribute(anchors_with_title[0], "title")
            away_title = safe_get_attribute(anchors_with_title[-1], "title")
            home_norm = normalize_team_name(home_title).lower()
            away_norm = normalize_team_name(away_title).lower()

            # Extract match ID
            link = row.locator("a[href*='spielbericht']").first
            href = safe_get_attribute(link, "href")
            mid = re.search(r"/spielbericht/(\d+)", href)
            if mid:
                match_id = mid.group(1)
                key = (home_norm, away_norm, match_date.strftime("%Y-%m-%d"))
                cache[key] = match_id
        except Exception:
            continue

    return cache


# ── Match ID lookup ───────────────────────────────────────────────────────────

def lookup_match_id(cache: dict, home: str, away: str, date: pd.Timestamp) -> str | None:
    """Find a match ID using flexible team name and ±1-day date matching."""
    home_norm = normalize_team_name(home).lower()
    away_norm = normalize_team_name(away).lower()

    for (cached_home, cached_away, cached_date_str), match_id in cache.items():
        try:
            cached_date = pd.to_datetime(cached_date_str)
        except Exception:
            continue

        date_ok = abs((cached_date - date).days) <= 1
        teams_ok = (
            (cached_home in home_norm or home_norm in cached_home or
             _token_match(cached_home, home_norm))
            and
            (cached_away in away_norm or away_norm in cached_away or
             _token_match(cached_away, away_norm))
        )
        if date_ok and teams_ok:
            return match_id
    return None


def _token_match(a: str, b: str) -> bool:
    """Check if any token from 'a' is in 'b' or vice versa (word-level matching)."""
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    return bool(a_tokens & b_tokens) and len(a_tokens & b_tokens) >= min(1, len(a_tokens) // 2)


# ── Lineup page scraper ───────────────────────────────────────────────────────

def scrape_lineup_from_page(page, match_id: str, home_team: str, away_team: str) -> list[dict]:
    """
    Given a Transfermarkt lineup page (already loaded), extract starting XI
    for both teams. Returns a list of player dicts.
    """
    players = []
    try:
        tables = page.locator("table.items").all()
        if len(tables) < 2:
            log.warning(f"  Only {len(tables)} lineup table(s) for match {match_id}")
            return players

        for idx, table in enumerate(tables[:2]):
            team_name = home_team if idx == 0 else away_team
            rows = table.locator("tbody > tr").all()

            for i in range(0, len(rows), 3):  # TM uses 3 rows per player in new layout
                if i >= len(rows):
                    break
                row = rows[i]
                cells = row.locator("td").all()
                if len(cells) < 5:
                    continue

                shirt_number = safe_inner_text(cells[0])
                name_anchor  = cells[3].locator("a").first
                player_name  = safe_inner_text(name_anchor)
                player_href  = safe_get_attribute(name_anchor, "href")
                position_raw = safe_inner_text(cells[4])
                position     = position_raw.split(",")[0].strip()

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
        log.error(f"  Error parsing lineup for {match_id}: {e}")

    return players


# ── Main scraping loop ─────────────────────────────────────────────────────────

def scrape_matches_with_cache(
    matches: pd.DataFrame,
    cache: dict,
    page,
    already_done: set,
    competition_label: str,
) -> list[dict]:
    """
    Scrape lineups for all matches in `matches` using pre-built `cache`.
    Skips already-done matches and logs progress incrementally.
    Returns list of new player records.
    """
    new_players = []
    buffer = []

    for _, row in matches.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        date = pd.to_datetime(row["date"])
        match_key = f"{date.strftime('%Y-%m-%d')}|{home}|{away}"

        if match_key in already_done:
            continue

        log.info(f"  [{competition_label}] {date.date()} {home} vs {away}")

        match_id = lookup_match_id(cache, home, away, date)
        if not match_id:
            log.warning(f"    → Not found in cache, skipping.")
            log_progress(match_key, None, 0)
            already_done.add(match_key)
            continue

        lineup_url = f"https://www.transfermarkt.com/x/aufstellung/spielbericht/{match_id}"
        try:
            page.goto(lineup_url, timeout=30000)
            dismiss_consent(page)
            page.wait_for_selector("table.items", timeout=10000)
        except (PWTimeout, Exception) as e:
            log.warning(f"    → Lineup page failed ({e}), skipping.")
            log_progress(match_key, match_id, 0)
            already_done.add(match_key)
            polite_sleep(2.0, 4.0)
            continue

        players = scrape_lineup_from_page(page, match_id, home, away)
        for p in players:
            p["match_date"]  = date.strftime("%Y-%m-%d")
            p["home_team"]   = home
            p["away_team"]   = away
            p["wc_cycle"]    = int(row.get("wc_cycle", 0))
            p["competition"] = competition_label

        buffer.extend(players)
        new_players.extend(players)
        log.info(f"    → {len(players)} players (match_id={match_id})")

        log_progress(match_key, match_id, len(players))
        already_done.add(match_key)

        # Incremental save
        if len(buffer) >= SAVE_EVERY_N * 22:  # ~22 players per match
            save_players(buffer)
            buffer = []
            log.info(f"    [Checkpoint] Saved {len(new_players)} player records so far.")

        polite_sleep()

    # Save remaining buffer
    if buffer:
        save_players(buffer)

    return new_players


def main():
    log.info("=" * 70)
    log.info("FULL COVERAGE LINEUP SCRAPER")
    log.info("=" * 70)

    # Create output dirs
    config.LINEUPS_DIR.mkdir(parents=True, exist_ok=True)

    # Load full scope
    log.info("\nLoading Gulati dataset and building match scope ...")
    df = pd.read_csv(config.GULATI_CSV)
    scope = get_full_match_scope(df)
    log.info(f"Total matches in scope: {len(scope)}")
    log.info(scope.groupby(["wc_cycle", "competition_group"]).size().to_string())

    # Load already-completed matches
    already_done = load_progress()
    log.info(f"\nAlready scraped: {len(already_done)} matches. Resuming ...")

    remaining = scope[~scope.apply(
        lambda r: f"{pd.to_datetime(r['date']).strftime('%Y-%m-%d')}|{r['home_team']}|{r['away_team']}"
        in already_done, axis=1
    )]
    log.info(f"Remaining to scrape: {len(remaining)} matches.")
    if len(remaining) == 0:
        log.info("All matches already scraped! Nothing to do.")
        return

    est_hours = len(remaining) * 3.5 / 3600
    log.info(f"Estimated time: {est_hours:.1f} hours (at 3.5s avg/match)")
    log.info("Starting scraper ... (safe to interrupt and resume)\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        matches_since_refresh = 0

        def maybe_refresh_browser():
            nonlocal context, page, matches_since_refresh
            if matches_since_refresh >= REFRESH_EVERY_N:
                log.info("  [Browser] Refreshing context to prevent memory leaks ...")
                try:
                    page.close()
                    context.close()
                except Exception:
                    pass
                context = browser.new_context(user_agent=USER_AGENT)
                page = context.new_page()
                matches_since_refresh = 0

        for wc_year in [2014, 2018, 2022]:
            log.info(f"\n{'='*60}")
            log.info(f"  WC CYCLE {wc_year}")
            log.info(f"{'='*60}")
            cycle_df = scope[scope["wc_cycle"] == wc_year]

            # ── World Cup final matches ────────────────────────────────────────
            wc_df = cycle_df[cycle_df["competition_group"] == "wc"]
            if len(wc_df) > 0:
                log.info(f"\n--- World Cup {wc_year} ({len(wc_df)} matches) ---")
                maybe_refresh_browser()
                cache = build_wc_schedule_cache(page, wc_year)
                polite_sleep(1, 2)
                if cache:
                    scrape_matches_with_cache(wc_df, cache, page, already_done, f"WC{wc_year}")
                    matches_since_refresh += len(wc_df)

            # ── Qualifiers per confederation ───────────────────────────────────
            qual_df = cycle_df[cycle_df["competition_group"] == "qualifier"]
            for confed, cycle_map in QUALIFIER_COMPS.items():
                if wc_year not in cycle_map:
                    continue
                comp_id, saison = cycle_map[wc_year]

                # Match qualifier rows to this confederation
                if confed == "Playoffs":
                    # Inter-confederation playoff matches are any qualifier
                    # not belonging to a single confederation
                    conf_mask = qual_df["tournament"].str.contains(
                        "playoff|Playoff|intercontinental", case=False, na=False
                    )
                else:
                    conf_mask = (
                        (qual_df["home_confed"] == confed) |
                        (qual_df["away_confed"] == confed)
                    )

                conf_df = qual_df[conf_mask]
                if len(conf_df) == 0:
                    continue

                log.info(f"\n--- {confed} Qualifiers {wc_year} ({len(conf_df)} matches) ---")
                maybe_refresh_browser()
                cache = build_qualifier_schedule_cache(page, comp_id, saison)
                polite_sleep(1, 2)
                if cache:
                    scrape_matches_with_cache(
                        conf_df, cache, page, already_done,
                        f"QUAL_{confed}_{wc_year}"
                    )
                    matches_since_refresh += len(conf_df)

            # ── Friendlies ─────────────────────────────────────────────────────
            fr_df = cycle_df[cycle_df["competition_group"] == "friendly"]
            if len(fr_df) > 0:
                # Group by year to build per-year friendly schedule caches
                # (Transfermarkt friendlies are organized by season/year)
                log.info(f"\n--- Friendlies pre-{wc_year} ({len(fr_df)} matches) ---")
                for fr_year in sorted(fr_df["date"].dt.year.unique()):
                    yr_df = fr_df[fr_df["date"].dt.year == fr_year]
                    if len(yr_df) == 0:
                        continue
                    log.info(f"  Friendlies {fr_year}: {len(yr_df)} matches")
                    maybe_refresh_browser()
                    cache = build_friendly_schedule_cache(page, fr_year)
                    polite_sleep(1, 2)
                    if cache:
                        scrape_matches_with_cache(
                            yr_df, cache, page, already_done,
                            f"FRIENDLY_{fr_year}"
                        )
                        matches_since_refresh += len(yr_df)

        try:
            page.close()
            context.close()
            browser.close()
        except Exception:
            pass

    # Final stats
    if PROGRESS_CSV.exists():
        prog = pd.read_csv(PROGRESS_CSV)
        n_found = len(prog[prog["match_id"] != ""])
        n_total = len(prog)
        total_players = prog["n_players"].sum()
        log.info(f"\n{'='*60}")
        log.info(f"SCRAPING COMPLETE")
        log.info(f"  Matches processed: {n_total}")
        log.info(f"  Match IDs found:   {n_found} ({100*n_found/max(n_total,1):.1f}%)")
        log.info(f"  Total players:     {int(total_players)}")
        log.info(f"  Output: {LINEUPS_CSV}")
        log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
