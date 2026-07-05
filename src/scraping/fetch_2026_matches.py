"""
fetch_2026_matches.py
Downloads 2026 international results from martj42/international_results,
filters to Jan 2026 onwards, normalizes team names to match Gulati dataset,
and saves to data/raw/matches_2026.csv.
"""
import sys
import csv
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MARTJ42_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Mapping from martj42 team names -> Gulati dataset team names
# Built by comparing unique team names in both datasets
TEAM_NAME_MAP = {
    "Korea Republic":           "South Korea",
    "Korea DPR":                "North Korea",
    "IR Iran":                  "Iran",
    "United States":            "United States",
    "C\u00f4te d'Ivoire":            "C\u00f4te d'Ivoire",
    "Cote d'Ivoire":            "C\u00f4te d'Ivoire",
    "Cape Verde Islands":       "Cabo Verde",
    "Cape Verde":               "Cabo Verde",
    "Ivory Coast":              "C\u00f4te d'Ivoire",
    "DR Congo":                 "DR Congo",
    "Bosnia-Herzegovina":       "Bosnia and Herzegovina",
    "Czechia":                  "Czechia",
    "Czech Republic":           "Czechia",
    "North Macedonia":          "North Macedonia",
    "Trinidad & Tobago":        "Trinidad and Tobago",
    "Antigua & Barbuda":        "Antigua and Barbuda",
    "St Kitts & Nevis":         "Saint Kitts and Nevis",
    "St Vincent & the Grenadines": "Saint Vincent and the Grenadines",
    "São Tomé & Príncipe":      "Sao Tome and Principe",
    "Chinese Taipei":           "Chinese Taipei",
    "China PR":                 "China",
    "Kyrgyz Republic":          "Kyrgyzstan",
    "Faroe Islands":            "Faroe Islands",
    "Guinea-Bissau":            "Guinea-Bissau",
    "Eswatini":                 "Swaziland",
    "St Lucia":                 "Saint Lucia",
    "Curacao":                  "Curaçao",
}


def normalize_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def main():
    out_raw  = Path("data/raw/matches_2026_raw.csv")
    out_norm = Path("data/raw/matches_2026.csv")
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MARTJ42_URL} ...")
    with urllib.request.urlopen(MARTJ42_URL) as response:
        content = response.read().decode("utf-8")

    lines = content.splitlines()
    reader = csv.DictReader(lines)
    all_rows = list(reader)
    print(f"Total rows in martj42: {len(all_rows)}")

    # Filter to 2026
    rows_2026 = [r for r in all_rows if r["date"] >= "2026-01-01"]
    print(f"Rows from 2026 onwards: {len(rows_2026)}")

    # Save raw (unnormalized)
    fieldnames = reader.fieldnames
    with open(out_raw, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_2026)
    print(f"Saved raw -> {out_raw}")

    # Normalize team names and save
    normalized = []
    for r in rows_2026:
        r2 = dict(r)
        r2["home_team"] = normalize_name(r["home_team"])
        r2["away_team"] = normalize_name(r["away_team"])
        normalized.append(r2)

    with open(out_norm, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)
    print(f"Saved normalized -> {out_norm}")

    # Print summary
    print("\nDate range:")
    dates = [r["date"] for r in rows_2026]
    print(f"  First: {min(dates)}")
    print(f"  Last:  {max(dates)}")

    tournaments = {}
    for r in rows_2026:
        t = r["tournament"]
        tournaments[t] = tournaments.get(t, 0) + 1
    print("\nMatch counts by tournament:")
    for t, c in sorted(tournaments.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}  {t}")

    # Show WC 2026 matches
    wc = [r for r in normalized if "FIFA World Cup" in r["tournament"] or "world cup" in r["tournament"].lower()]
    print(f"\nFIFA World Cup 2026 matches: {len(wc)}")
    for r in wc[-10:]:
        print(f"  {r['date']}  {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']}  [{r['tournament']}]")


if __name__ == "__main__":
    main()
