"""
scrape_player_stats.py — Scrape per-season club statistics for every player
found in the scraped lineups file.

For each unique (player_id, wc_cycle) pair, fetches the player's stats
from the club season immediately preceding that World Cup.

Output: data/raw/player_stats/all_player_stats.csv
"""

import sys
import re
import logging
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.scraping.utils import (
    polite_sleep, new_browser_context,
    safe_inner_text, safe_get_attribute, log,
)

# ── Output ────────────────────────────────────────────────────────────────────
PLAYER_STATS_CSV = config.PLAYER_STATS_CSV

# Transfermarkt player stats URL pattern
# e.g. https://www.transfermarkt.com/lionel-messi/leistungsdatendetails/spieler/28003/plus/0?saison=2021
TM_PLAYER_STATS_URL = (
    "https://www.transfermarkt.com/{slug}/leistungsdatendetails"
    "/spieler/{player_id}/plus/0?saison={saison}"
)

# Fallback slug when we don't have the player's name slug
DEFAULT_SLUG = "x"


def determine_wc_cycle(match_date: pd.Timestamp) -> int | None:
    """Map a match date to the WC cycle year it belongs to (2014/2018/2022)."""
    year = match_date.year
    if 2012 <= year <= 2014:
        return 2014
    elif 2016 <= year <= 2018:
        return 2018
    elif 2020 <= year <= 2022:
        return 2022
    elif 2024 <= year <= 2026:
        return 2026
    return None


def extract_player_slug(name: str) -> str:
    """Convert a player name to a Transfermarkt URL slug."""
    import unicodedata
    # Normalize unicode, lowercase, replace spaces with hyphens
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s-]", "", name)
    name = re.sub(r"\s+", "-", name)
    return name


def parse_stats_table(page) -> dict:
    """
    Parse the Transfermarkt player performance detail page.
    Returns a dict of aggregated season stats across all competitions.
    """
    stats = {
        "appearances":    0,
        "goals":          0,
        "assists":        0,
        "minutes_played": 0,
        "yellow_cards":   0,
        "red_cards":      0,
    }

    try:
        # Each competition row in the stats table
        rows = page.locator("table.items tbody tr").all()
        for row in rows:
            cells = row.locator("td").all()
            if len(cells) < 7:
                continue

            def to_int(idx):
                txt = safe_inner_text(cells[idx]).replace(".", "").replace("'", "").strip()
                if txt in ("-", "", "–"):
                    return 0
                try:
                    return int(txt)
                except ValueError:
                    return 0

            # Transfermarkt column order (varies slightly; use robust parsing):
            # col 0: competition name
            # col 1: appearances
            # col 2: goals
            # col 3: assists
            # col 4: own goals (ignore)
            # col 5: yellow cards
            # col 6: yellow-red (count as red for simplicity)
            # col 7: red cards
            # col 8: minutes played
            try:
                stats["appearances"]    += to_int(1)
                stats["goals"]          += to_int(2)
                stats["assists"]        += to_int(3)
                stats["yellow_cards"]   += to_int(5)
                stats["red_cards"]      += to_int(6) + to_int(7)
                # Minutes: strip "'" suffix
                min_txt = safe_inner_text(cells[8]).replace("'", "").replace(".", "").strip()
                if min_txt not in ("-", "", "–"):
                    try:
                        stats["minutes_played"] += int(min_txt)
                    except ValueError:
                        pass
            except IndexError:
                continue

    except Exception as e:
        log.debug(f"  Stats table parse error: {e}")

    return stats


def scrape_player(page, player_id: str, player_name: str,
                  saison: str, wc_cycle: int) -> dict | None:
    """
    Scrape stats for one player for one season. Returns a dict or None.
    """
    slug = extract_player_slug(player_name)
    url = TM_PLAYER_STATS_URL.format(slug=slug, player_id=player_id, saison=saison)

    try:
        page.goto(url, timeout=30000)
        # Wait for either the stats table or the profile header
        page.wait_for_selector("div.dataArea, h1.data-header__headline-wrapper", timeout=10000)
    except (PWTimeout, Exception) as e:
        log.warning(f"  Page load failed for player {player_id} saison {saison}: {e}")
        return None

    # Try with default slug if the pretty slug didn't work
    if "404" in page.title() or "Error" in page.title():
        url_fallback = TM_PLAYER_STATS_URL.format(
            slug=DEFAULT_SLUG, player_id=player_id, saison=saison
        )
        try:
            page.goto(url_fallback, timeout=20000)
            page.wait_for_selector("div.dataArea", timeout=8000)
        except Exception:
            log.warning(f"  Fallback also failed for player {player_id}.")
            return None

    # Extract date of birth
    dob = ""
    try:
        dob_el = page.locator("span[itemprop='birthDate']")
        if dob_el.count() > 0:
            dob = safe_inner_text(dob_el.first)
    except Exception:
        pass

    # Extract current club name for that season
    club = ""
    try:
        club_el = page.locator("a.hauptlink").first
        club = safe_inner_text(club_el)
    except Exception:
        pass

    stats = parse_stats_table(page)

    return {
        "player_id":      player_id,
        "player_name":    player_name,
        "wc_cycle":       wc_cycle,
        "saison":         saison,
        "club":           club,
        "date_of_birth":  dob,
        **stats,
    }


def main():
    if not config.LINEUPS_CSV.exists():
        log.error(f"Lineups file not found: {config.LINEUPS_CSV}")
        log.error("Run scrape_lineups.py first.")
        sys.exit(1)

    log.info("Loading lineups …")
    lineups_df = pd.read_csv(config.LINEUPS_CSV, parse_dates=["match_date"])

    # Determine WC cycle for each lineup row
    lineups_df["wc_cycle"] = lineups_df["match_date"].apply(determine_wc_cycle)
    lineups_df = lineups_df.dropna(subset=["wc_cycle", "player_id"])
    lineups_df["wc_cycle"] = lineups_df["wc_cycle"].astype(int)

    # Build unique (player_id, player_name, wc_cycle) pairs
    unique_pairs = (
        lineups_df[["player_id", "player_name", "wc_cycle"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    log.info(f"  → {len(unique_pairs)} unique (player, WC cycle) pairs to scrape.")

    # Resume from existing output if present
    existing_ids: set = set()
    if config.PLAYER_STATS_CSV.exists():
        existing_df = pd.read_csv(config.PLAYER_STATS_CSV)
        existing_ids = set(
            zip(existing_df["player_id"].astype(str), existing_df["wc_cycle"].astype(str))
        )
        log.info(f"  → Resuming: {len(existing_ids)} already scraped.")

    config.PLAYER_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    with sync_playwright() as p:
        browser, context, page = new_browser_context(p)

        for i, row in unique_pairs.iterrows():
            pid       = str(int(row["player_id"]))
            name      = str(row["player_name"])
            wc_cycle  = int(row["wc_cycle"])
            saison    = config.SEASON_MAPPING.get(wc_cycle, str(wc_cycle - 1))

            key = (pid, str(wc_cycle))
            if key in existing_ids:
                continue

            log.info(f"[{i+1}/{len(unique_pairs)}] {name} (ID: {pid}) — saison {saison}")
            record = scrape_player(page, pid, name, saison, wc_cycle)

            if record:
                results.append(record)
                log.info(f"  → {record['appearances']} apps, {record['goals']} goals, "
                         f"{record['minutes_played']} min")

            # Save incrementally every 50 players
            if (i + 1) % 50 == 0 and results:
                _save_results(results, append=config.PLAYER_STATS_CSV.exists())
                results = []
                log.info(f"  [Checkpoint] Saved to {config.PLAYER_STATS_CSV}")

            polite_sleep()

        browser.close()

    # Final save
    if results:
        _save_results(results, append=config.PLAYER_STATS_CSV.exists())

    log.info(f"\n✓ Done. Stats saved to {config.PLAYER_STATS_CSV}")


def _save_results(results: list[dict], append: bool) -> None:
    df = pd.DataFrame(results)
    if append:
        df.to_csv(config.PLAYER_STATS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(config.PLAYER_STATS_CSV, index=False)


if __name__ == "__main__":
    main()
