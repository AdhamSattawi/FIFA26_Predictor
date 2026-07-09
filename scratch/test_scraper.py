import sys
from pathlib import Path
import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.scraping.utils import new_browser_context, dismiss_consent
from src.scraping.scrape_lineups import find_match_id_on_tm, scrape_lineup_from_page

def main():
    print("Testing updated lineup parser for match 2384328...")
    with sync_playwright() as p:
        browser, context, page = new_browser_context(p)
        
        match_id = "2384328"
        lineup_url = f"https://www.transfermarkt.com/x/aufstellung/spielbericht/{match_id}"
        print(f"Loading lineups from: {lineup_url}")
        page.goto(lineup_url)
        dismiss_consent(page)
        page.wait_for_timeout(3000)
        
        players = scrape_lineup_from_page(page, match_id, "Brazil", "Croatia")
        print(f"Scraped {len(players)} players.")
        for p_rec in players:
            print(f"  {p_rec['team']} | #{p_rec['shirt_number']} {p_rec['position']}: {p_rec['player_name']} (ID: {p_rec['player_id']})")
            
        browser.close()

if __name__ == "__main__":
    main()
