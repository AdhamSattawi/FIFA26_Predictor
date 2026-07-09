import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scraping.utils import new_browser_context, dismiss_consent, safe_inner_text
from src.scraping.scrape_player_stats import scrape_player

def main():
    print("Testing player stats scraper for Thiago Silva (ID: 29241, saison: 2013)...")
    with sync_playwright() as p:
        browser, context, page = new_browser_context(p)
        
        player_id = "29241"
        player_name = "Thiago Silva"
        saison = "2013"
        wc_cycle = 2014
        
        res = scrape_player(page, player_id, player_name, saison, wc_cycle)
        print("Result details:")
        if res:
            for k, v in res.items():
                print(f"  {k}: {v}")
        else:
            print("FAILED to scrape player stats.")
            
        browser.close()

if __name__ == "__main__":
    main()
