import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict


def simulate_game(p_home_win: float) -> int:
    """Return 1 if home wins, 0 if away wins."""
    return int(np.random.rand() < p_home_win)


def simulate_one_regular_season(
    schedule: pd.DataFrame,
    model,
    feature_cols: List[str],
) -> Dict[str, int]:
    """
    Simulate ONE regular season.

    schedule: DataFrame with at least:
      - home_team, away_team
      - feature_cols (e.g. pd_diff, pf_diff, pa_diff)
    """
    teams = pd.unique(schedule[["home_team", "away_team"]].values.ravel("K"))
    wins = {team: 0 for team in teams}

    X = schedule[feature_cols]
    p_home = model.predict_proba(X)[:, 1]

    for (_, row), p in zip(schedule.iterrows(), p_home):
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_win_sim = simulate_game(p)

        if home_win_sim == 1:
            wins[home_team] += 1
        else:
            wins[away_team] += 1

    return wins


def simulate_regular_season(
    schedule: pd.DataFrame,
    model,
    feature_cols: List[str],
    n_sims: int = 100,
) -> pd.DataFrame:
    """
    Run many simulated seasons and summarize average wins.
    """
    teams = pd.unique(schedule[["home_team", "away_team"]].values.ravel("K"))
    results = defaultdict(list)

    for _ in range(n_sims):
        wins = simulate_one_regular_season(schedule, model, feature_cols)
        for team in teams:
            results[team].append(wins.get(team, 0))

    summary = []
    for team in teams:
        w = np.array(results[team])
        summary.append(
            {
                "team": team,
                "mean_wins": float(w.mean()),
                "median_wins": float(np.median(w)),
            }
        )

    return pd.DataFrame(summary).sort_values("mean_wins", ascending=False)
