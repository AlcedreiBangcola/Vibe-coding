import pandas as pd
from pathlib import Path
import time


def fetch_season_games(season: int) -> pd.DataFrame:
    """
    Fetch one NBA season’s regular-season games from Basketball-Reference.

    season: the year the season ends, e.g. 2024 for 2023–24.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    print(f"Fetching: {url}")
    df_list = pd.read_html(url)

    games = df_list[0]
    # Drop the "Playoffs" separator rows if present
    games = games[games["Date"] != "Playoffs"]
    games["Season"] = season
    return games


def fetch_multiple_seasons(start: int, end: int) -> pd.DataFrame:
    """
    Fetch multiple seasons from start to end (inclusive).
    Example: fetch_multiple_seasons(2015, 2024)
    """
    all_games = []
    for season in range(start, end + 1):
        print(f"Season {season}")
        df = fetch_season_games(season)
        all_games.append(df)
        time.sleep(1.0)  # be nice to the site

    return pd.concat(all_games, ignore_index=True)


def main():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df = fetch_multiple_seasons(2015, 2024)
    out_path = Path("data/raw/games_2015_2024.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved games to {out_path}")


if __name__ == "__main__":
    main()
