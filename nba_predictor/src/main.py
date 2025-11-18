from pathlib import Path

from joblib import load

from src.build_dataset import prepare_games, build_team_features, make_model_dataset
from src.simulate_season import simulate_regular_season
from src.simulate_playoffs import simulate_championships


def main():
    # 1. Load model
    model = load("models/game_model.joblib")
    feature_cols = load("models/feature_cols.joblib")

    # 2. Use the last season as a schedule to test simulation
    raw_path = Path("data/raw/games_2015_2024.csv")
    games = prepare_games(str(raw_path))

    last_season = games["Season"].max()

    # Build rolling features for all games, then filter last season
    team_games = build_team_features(games)
    dataset = make_model_dataset(games, team_games)
    schedule_features = dataset[dataset["Season"] == last_season].copy()

    print(f"Simulating regular season {last_season} over 200 runs...")
    season_summary = simulate_regular_season(
        schedule_features, model, feature_cols, n_sims=200
    )
    print(season_summary.head(10))

    print(f"\nSimulating full season + playoffs {last_season} for title odds...")
    title_odds = simulate_championships(
        schedule_features, model, feature_cols, n_sims=300
    )
    print(title_odds.head(10))


if __name__ == "__main__":
    main()
