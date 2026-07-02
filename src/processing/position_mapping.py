"""
position_mapping.py — Maps Transfermarkt position strings to the
canonical 11-slot ordering used as CNN input.

Canonical order (GK → ST, defense → attack, right → left):
  0  GK   — Goalkeeper
  1  RB   — Right-Back / Right Wing-Back
  2  CB1  — Centre-Back (right)
  3  CB2  — Centre-Back (left)
  4  LB   — Left-Back / Left Wing-Back
  5  CDM  — Defensive Midfielder
  6  CM   — Central Midfielder
  7  CAM  — Attacking Midfielder
  8  RW   — Right Winger / Right Midfielder
  9  LW   — Left Winger / Left Midfielder
  10 ST   — Striker / Centre-Forward
"""

from config import POSITION_ORDER, N_PLAYERS

# ── Raw Transfermarkt position → canonical slot label ─────────────────────────
# Keys are lowercase versions of what Transfermarkt returns.
POSITION_MAP: dict[str, str] = {
    # Goalkeepers
    "goalkeeper":                "GK",
    "torwart":                   "GK",
    "gk":                        "GK",

    # Right back
    "right-back":                "RB",
    "right back":                "RB",
    "rechter verteidiger":       "RB",
    "right wing-back":           "RB",
    "rb":                        "RB",
    "rwb":                       "RB",

    # Centre backs
    "centre-back":               "CB",  # will be split into CB1/CB2
    "center-back":               "CB",
    "central defender":          "CB",
    "innenverteidiger":          "CB",
    "cb":                        "CB",
    "centreback":                "CB",

    # Left back
    "left-back":                 "LB",
    "left back":                 "LB",
    "linker verteidiger":        "LB",
    "left wing-back":            "LB",
    "lb":                        "LB",
    "lwb":                       "LB",

    # Defensive midfielder
    "defensive midfield":        "CDM",
    "defensive midfielder":      "CDM",
    "defensive midfielders":     "CDM",
    "defensives mittelfeld":     "CDM",
    "holding midfielder":        "CDM",
    "cdm":                       "CDM",
    "dm":                        "CDM",

    # Central midfielder
    "central midfield":          "CM",
    "central midfielder":        "CM",
    "zentrales mittelfeld":      "CM",
    "cm":                        "CM",
    "box-to-box midfielder":     "CM",

    # Attacking midfielder
    "attacking midfield":        "CAM",
    "attacking midfielder":      "CAM",
    "offensives mittelfeld":     "CAM",
    "second striker":            "CAM",
    "cam":                       "CAM",
    "am":                        "CAM",
    "shadow striker":            "CAM",

    # Right winger
    "right winger":              "RW",
    "right midfield":            "RW",
    "rechtes mittelfeld":        "RW",
    "rw":                        "RW",
    "rm":                        "RW",

    # Left winger
    "left winger":               "LW",
    "left midfield":             "LW",
    "linkes mittelfeld":         "LW",
    "lw":                        "LW",
    "lm":                        "LW",

    # Striker
    "centre-forward":            "ST",
    "center-forward":            "ST",
    "striker":                   "ST",
    "mittelstürmer":             "ST",
    "st":                        "ST",
    "cf":                        "ST",
    "fw":                        "ST",
    "forward":                   "ST",
}

# Priority order for filling slots when multiple players map to the same slot
SLOT_PRIORITY = {
    "GK":  0,
    "RB":  1,
    "CB":  2,   # CB1 + CB2 share priority
    "LB":  4,
    "CDM": 5,
    "CM":  6,
    "CAM": 7,
    "RW":  8,
    "LW":  9,
    "ST":  10,
}


def map_position(raw_position: str) -> str:
    """Map a raw Transfermarkt position string to a canonical slot label."""
    return POSITION_MAP.get(raw_position.lower().strip(), "CM")  # default to CM


def assign_slots(players: list[dict]) -> list[dict]:
    """
    Given a list of player dicts (each with a 'position' field already mapped
    to canonical labels like 'GK', 'CB', 'RB', etc.), assign them to the
    11 canonical position slots.

    Returns a list of 11 player dicts (in slot order 0–10), with missing
    slots filled with a zero vector marker (player_id = None).
    """
    # Separate CBs for CB1/CB2 split
    slots: dict[str, list[dict]] = {pos: [] for pos in POSITION_ORDER}
    cb_pool: list[dict] = []

    for p in players:
        pos = map_position(p.get("position", ""))
        if pos == "CB":
            cb_pool.append(p)
        elif pos in slots:
            slots[pos].append(p)

    # Assign CBs to CB1 and CB2
    if cb_pool:
        slots["CB1"].append(cb_pool[0]) if "CB1" in slots else None
        if len(cb_pool) > 1:
            slots["CB2"].append(cb_pool[1]) if "CB2" in slots else None

    # Build the ordered 11-slot list
    result = []
    for slot in POSITION_ORDER:
        if slots.get(slot):
            # Take the first player assigned to this slot
            player = slots[slot][0].copy()
            player["canonical_slot"] = slot
            result.append(player)
        else:
            # Empty slot — will be filled with median values during feature engineering
            result.append({
                "canonical_slot": slot,
                "player_id":      None,
                "player_name":    f"[MISSING {slot}]",
                "position":       slot,
            })

    return result  # length 11


def lineup_to_slots(lineup_df: pd.DataFrame, team: str) -> list[dict]:
    """
    Given a DataFrame of lineup rows for one match, filter to one team
    and assign the 11 canonical slots.

    Returns a list of 11 dicts (one per slot).
    """
    import pandas as pd
    team_df = lineup_df[lineup_df["team"] == team]
    players = team_df.to_dict("records")
    return assign_slots(players)


# Allow running as a standalone check
if __name__ == "__main__":
    import pandas as pd
    print("Position map loaded:", len(POSITION_MAP), "entries")
    print("Canonical slots:", POSITION_ORDER)

    # Quick test
    test_players = [
        {"position": "Goalkeeper",      "player_name": "A", "player_id": "1"},
        {"position": "Right-Back",      "player_name": "B", "player_id": "2"},
        {"position": "Centre-Back",     "player_name": "C", "player_id": "3"},
        {"position": "Centre-Back",     "player_name": "D", "player_id": "4"},
        {"position": "Left-Back",       "player_name": "E", "player_id": "5"},
        {"position": "Defensive Midfield", "player_name": "F", "player_id": "6"},
        {"position": "Central Midfield",  "player_name": "G", "player_id": "7"},
        {"position": "Attacking Midfield","player_name": "H", "player_id": "8"},
        {"position": "Right Winger",    "player_name": "I", "player_id": "9"},
        {"position": "Left Winger",     "player_name": "J", "player_id": "10"},
        {"position": "Centre-Forward",  "player_name": "K", "player_id": "11"},
    ]
    result = assign_slots(test_players)
    for r in result:
        print(f"  Slot {r['canonical_slot']:4s} → {r['player_name']}")
