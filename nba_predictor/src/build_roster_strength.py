import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# Map Basketball-Reference team abbreviations to full team names
TEAM_ABBR_TO_NAME: Dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BRK": "Brooklyn Nets",
    "CHI": "Chicago Bulls",
    "CHO": "Charlotte Hornets",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def fetch_player_advanced(season: int) -> pd.DataFrame:
    """
    Fetch per-player advanced stats from Basketball-Reference.

    season = year the season ends, e.g. 2024 for 2023–24, 2025 for 2024–25.
    Handles various table header formats.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    print(f"Fetching player advanced stats from {url}")
    tables = pd.read_html(url, header=0)
    df = tables[0]

    # If the table has multi-level columns, flatten them to the last level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Drop repeated header rows inside the table, if present
    if "Player" in df.columns:
        df = df[df["Player"] != "Player"]

    # Detect the team column
    team_col = None
    for cand in ["Tm", "Team", "Tm.1", "Team.1"]:
        if cand in df.columns:
            team_col = cand
            break

    if team_col is None:
        raise ValueError(
            f"Could not find team column in advanced stats table. "
            f"Got columns: {list(df.columns)}"
        )

    # Remove "TOT" rows (overall totals across teams; we want per-team contributions)
    df = df[df[team_col] != "TOT"]

    # Ensure we have VORP and minutes (MP)
    if "VORP" not in df.columns:
        raise ValueError(
            f"Could not find VORP column in advanced stats table. "
            f"Got columns: {list(df.columns)}"
        )
    if "MP" not in df.columns:
        raise ValueError(
            f"Could not find MP (minutes) column in advanced stats table. "
            f"Got columns: {list(df.columns)}"
        )

    df["VORP"] = pd.to_numeric(df["VORP"], errors="coerce").fillna(0)
    df["MP"] = pd.to_numeric(df["MP"], errors="coerce").fillna(0)

    # Normalize team column name to 'Tm' for the rest of our code
    df = df.rename(columns={team_col: "Tm"})

    return df


def build_team_roster_strength(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Aggregate player value to team-level 'roster_strength'.

    We use a simple metric:
        player_value = VORP * MP
    and sum the top N players per team.
    """
    df = df.copy()
    df["player_value"] = df["VORP"] * df["MP"]

    # For each team, keep top N players by player_value
    df_sorted = df.sort_values(["Tm", "player_value"], ascending=[True, False])
    df_top = df_sorted.groupby("Tm").head(top_n)

    team_value = df_top.groupby("Tm")["player_value"].sum().reset_index()

    # Map abbreviations to full names
    team_value["team"] = team_value["Tm"].map(TEAM_ABBR_TO_NAME)

    # Drop any teams we don't have mapping for
    team_value = team_value.dropna(subset=["team"])

    team_value = team_value[["team", "player_value"]].rename(
        columns={"player_value": "roster_strength"}
    )

    # Sort just for nicer viewing
    team_value = team_value.sort_values("roster_strength", ascending=False).reset_index(
        drop=True
    )

    return team_value


def main():
    """
    Usage:
        python -m src.build_roster_strength 2025

    This will:
      - Download player advanced stats for 2024–25 season (NBA_2025_advanced.html)
      - Compute team roster strengths from top N players
      - Save to data/processed/team_roster_strength.csv
    """
    if len(sys.argv) >= 2:
        try:
            season = int(sys.argv[1])
        except ValueError:
            print("Season must be an integer, e.g. 2025")
            sys.exit(1)
    else:
        season = 2024

    df_advanced = fetch_player_advanced(season)
    team_strength = build_team_roster_strength(df_advanced, top_n=8)

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "team_roster_strength.csv"
    team_strength.to_csv(out_path, index=False)

    print(f"Saved team roster strength to {out_path}")
    print(team_strength.head(10))


if __name__ == "__main__":
    main()
