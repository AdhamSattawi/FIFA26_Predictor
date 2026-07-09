import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.scraping.scrape_lineups import get_match_scope

df = pd.read_csv(config.GULATI_CSV)
scope_df = get_match_scope(df)
print(f"Total matches in scope: {len(scope_df)}")
print(f"World Cup matches: {len(scope_df[scope_df['is_world_cup'] == 1])}")
print(f"Qualifier matches: {len(scope_df[scope_df['is_qualifier'] == 1])}")
print(f"Friendly matches: {len(scope_df[scope_df['is_friendly'] == 1])}")
