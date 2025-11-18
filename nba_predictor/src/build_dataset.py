import pandas as pd
import numpy as np
from pathlib import Path


def prepare_games(path: str) -> pd.DataFrame:
    games = pd.read_csv(path)

    games = games.rename(
        columns={
            "Visitor/Neutral": "away_team",
            "Home/Neutral": "home_team",
            "PTS": "away_pts",
            "PTS.1": "home_pts",
        }
    )

    # Drop unfinished games
    games = games.dropna(subset=["home_pts", "away_pts"])

    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
    games["Date"] = pd.to_datetime(games["Date"])
    games = games.sort_values("Date")

    return games


def build_team_features(games: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Build per-team rolling stats + rest information.
    window: how many games to use for rolling averages (larger = smoother).
    """
    # Row per team per game
    home = games[
        ["Date", "Season", "home_team", "home_pts", "away_pts"]
    ].rename(
        columns={
            "home_team": "team",
            "home_pts": "pts_for",
            "away_pts": "pts_against",
        }
    )
    away = games[
        ["Date", "Season", "away_team", "away_pts", "home_pts"]
    ].rename(
        columns={
            "away_team": "team",
            "away_pts": "pts_for",
            "home_pts": "pts_against",
        }
    )

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(["team", "Date"])

    team_games["point_diff"] = team_games["pts_for"] - team_games["pts_against"]

    # Rolling stats using only past games (shift)
    team_games["rolling_pd"] = (
        team_games.groupby("team")["point_diff"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    team_games["rolling_pf"] = (
        team_games.groupby("team")["pts_for"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    team_games["rolling_pa"] = (
        team_games.groupby("team")["pts_against"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # --- New: rest & back-to-back info ---
    team_games["days_since_last"] = team_games.groupby("team")["Date"].diff().dt.days
    # First game: treat as "well-rested" 3 days if NaN
    team_games["days_since_last"] = team_games["days_since_last"].fillna(3)

    # Back-to-back flag: 0 or 1
    team_games["is_b2b"] = (team_games["days_since_last"] <= 1).astype(int)

    return team_games


def make_model_dataset(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    tg = team_games[
        [
            "Date",
            "team",
            "rolling_pd",
            "rolling_pf",
            "rolling_pa",
            "days_since_last",
            "is_b2b",
        ]
    ]

    home_feats = tg.rename(
        columns={
            "team": "home_team",
            "rolling_pd": "home_rolling_pd",
            "rolling_pf": "home_rolling_pf",
            "rolling_pa": "home_rolling_pa",
            "days_since_last": "home_days_since_last",
            "is_b2b": "home_is_b2b",
        }
    )

    away_feats = tg.rename(
        columns={
            "team": "away_team",
            "rolling_pd": "away_rolling_pd",
            "rolling_pf": "away_rolling_pf",
            "rolling_pa": "away_rolling_pa",
            "days_since_last": "away_days_since_last",
            "is_b2b": "away_is_b2b",
        }
    )

    df = games.merge(home_feats, on=["Date", "home_team"], how="left")
    df = df.merge(away_feats, on=["Date", "away_team"], how="left")

    # Feature differences
    df["pd_diff"] = df["home_rolling_pd"] - df["away_rolling_pd"]
    df["pf_diff"] = df["home_rolling_pf"] - df["away_rolling_pf"]
    df["pa_diff"] = df["home_rolling_pa"] - df["away_rolling_pa"]

    # New: rest/back-to-back features
    df["rest_diff"] = df["home_days_since_last"] - df["away_days_since_last"]
    df["b2b_diff"] = df["home_is_b2b"] - df["away_is_b2b"]

    df = df.dropna(
        subset=[
            "pd_diff",
            "pf_diff",
            "pa_diff",
            "rest_diff",
            "b2b_diff",
        ]
    )

    return df


def main():
    raw_path = Path("data/raw/games_2015_2024.csv")
    games = prepare_games(str(raw_path))
    team_games = build_team_features(games)
    dataset = make_model_dataset(games, team_games)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out_path = Path("data/processed/model_dataset.csv")
    dataset.to_csv(out_path, index=False)
    print(f"Saved model dataset to {out_path}")


if __name__ == "__main__":
    main()
