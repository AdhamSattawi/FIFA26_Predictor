"""
utils.py — Shared scraping utilities for Transfermarkt.
"""

import random
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Transfermarkt base URL
TM_BASE = "https://www.transfermarkt.com"

# Browser user-agent to avoid blocks
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Polite delay range between requests (seconds)
DELAY_MIN = 2.5
DELAY_MAX = 5.0


def polite_sleep(min_s: float = DELAY_MIN, max_s: float = DELAY_MAX) -> None:
    """Sleep a random amount to avoid hammering the server."""
    delay = random.uniform(min_s, max_s)
    time.sleep(delay)


def new_browser_context(playwright):
    """
    Launch a Chromium browser context with a realistic user-agent.
    Returns (browser, context, page).
    """
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(user_agent=USER_AGENT)
    page = context.new_page()
    return browser, context, page


def dismiss_consent(page) -> None:
    """Dismiss GDPR / cookie consent dialogs on Transfermarkt if present."""
    try:
        # Check main page
        accept_btn = page.locator("button:has-text('Accept & continue'), button:has-text('Accept'), button:has-text('Zustimmen')")
        if accept_btn.count() > 0:
            accept_btn.first.click(timeout=2000)
            page.wait_for_timeout(1000)
            return
        
        # Check if inside frame
        for frame in page.frames:
            try:
                frame_btn = frame.locator("button:has-text('Accept & continue'), button:has-text('Akzeptieren'), button:has-text('Zustimmen')")
                if frame_btn.count() > 0:
                    frame_btn.first.click(timeout=2000)
                    page.wait_for_timeout(1000)
                    return
            except Exception:
                pass
    except Exception:
        pass


def safe_inner_text(locator, default: str = "") -> str:
    """Extract inner text from a locator, returning default on failure."""
    try:
        return locator.inner_text(timeout=3000).strip()
    except Exception:
        return default


def safe_get_attribute(locator, attr: str, default: str = "") -> str:
    """Get an attribute from a locator, returning default on failure."""
    try:
        val = locator.get_attribute(attr, timeout=3000)
        return val.strip() if val else default
    except Exception:
        return default


# ── Team name normalisation ───────────────────────────────────────────────────
# Maps Transfermarkt team names → Gulati dataset team names
# (populated with known discrepancies; extend as needed)
TM_TO_GULATI: dict[str, str] = {
    "Korea Republic":       "South Korea",
    "United States":        "United States",
    "USA":                  "United States",
    "Ivory Coast":          "Cote d'Ivoire",
    "Côte d'Ivoire":        "Cote d'Ivoire",
    "Czech Republic":       "Czech Republic",
    "Czechia":              "Czech Republic",
    "Bosnia-Herzegovina":   "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "China PR":             "China",
    "Chinese Taipei":       "Taiwan",
    "North Macedonia":      "North Macedonia",
    "FYR Macedonia":        "North Macedonia",
    "Cape Verde":           "Cape Verde",
    "Cabo Verde":           "Cape Verde",
    "Guinea Bissau":        "Guinea-Bissau",
    "São Tomé e Príncipe":  "Sao Tome and Principe",
    "Türkiye":              "Turkey",
    "Trinidad & Tobago":    "Trinidad and Tobago",
    "St. Kitts & Nevis":    "Saint Kitts and Nevis",
    "St. Vincent & Grenadines": "Saint Vincent and the Grenadines",
    "Antigua & Barbuda":    "Antigua and Barbuda",
    "Congo DR":             "DR Congo",
    "DR Congo":             "DR Congo",
    "Congo DRC":            "DR Congo",
}


def normalize_team_name(name: str) -> str:
    """Normalise a Transfermarkt team name to match the Gulati dataset."""
    return TM_TO_GULATI.get(name, name)
