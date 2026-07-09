import asyncio
import sys
from pathlib import Path
import pandas as pd
from playwright.async_api import async_playwright

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scraping.utils import new_browser_context, dismiss_consent
# Since we need to test the sync version, let's just write a simple sync script
# or use playwright.sync_api in a normal python execution.
