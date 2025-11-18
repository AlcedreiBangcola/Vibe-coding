import math
import random
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd

from .simulate_season import simulate_one_regular_season

# Very simple conference mapping (must match names in your data)
TEAM_CONFERENCE: Dict[str, str] = {
    # East
    "Atlanta Hawks": "East",
    "Boston Celtics": "East",
    "Brooklyn Nets": "East",
    "Charlotte Hornets": "East",
    "Chicago Bulls": "East",
    "Cleveland Cavaliers": "East",
    "Detroit Pistons": "East",
    "Indiana Pacers": "East",
    "Miami Heat": "East",
    "Milwaukee Bucks": "East",
    "New York Knicks": "East",
    "Orlando Magic": "East",
    "Philadelphia 76ers": "East",
    "Toronto Raptors": "East",
    "Washington Wizards": "East",
    # West
    "Dallas Mavericks": "West",
    "Denver Nuggets": "West",
    "Golden State Warriors": "West",
    "Houston Rockets": "West",
    "Los Angeles Clippers": "West",
    "Los Angeles Lakers": "West",
    "Memphis Grizzlies": "West",
    "Minnesota Timberwolves": "West",
    "New Orleans Pelicans": "West",
    "Oklahoma City Thunder": "West",
    "Phoenix Suns": "West",
    "Portland Trail Blazers": "West",
    "Sacramento Kings": "West",
    "San Antonio Spurs": "West",
    "Utah Jazz": "West",
}


def build_ratings(wins: Dict[str, int]) -> Dict[str, float]:
    """
    Simple team rating based on simulated regular-season wins.
    """
    return {team: float(w) for team, w in wins.items()}


def game_home_win_prob(
    home_team: str,
    away_team: str,
    ratings: Dict[str, float],
    k: float = 0.08,
    home_adv: float = 0.05,
) -> float:
    """
    Compute probability that the HOME team wins a playoff game,
    based on rating difference + home-court bump.
    """
    r_home = ratings.get(home_team, 0.0)
    r_away = ratings.get(away_team, 0.0)

    diff = r_home - r_away
    neutral = 1.0 / (1.0 + math.exp(-k * diff))  # logistic on rating diff

    # Add simple home advantage and clamp
    p_home = neutral + home_adv
    p_home = max(0.05, min(0.95, p_home))
    return p_home


HOME_PATTERN = ["high", "high", "low", "low", "high", "low", "high"]  # 2-2-1-1-1


def simulate_series(
    high_seed_team: str,
    low_seed_team: str,
    ratings: Dict[str, float],
    best_of: int = 7,
) -> str:
    """
    Simulate a best-of-7 series where high_seed_team has home court.
    """
    wins = {high_seed_team: 0, low_seed_team: 0}
    needed = (best_of // 2) + 1

    game_index = 0
    while wins[high_seed_team] < needed and wins[low_seed_team] < needed:
        loc = HOME_PATTERN[game_index]
        if loc == "high":
            home, away = high_seed_team, low_seed_team
        else:
            home, away = low_seed_team, high_seed_team

        p_home = game_home_win_prob(home, away, ratings)
        home_wins = random.random() < p_home

        if home_wins:
            wins[home] += 1
        else:
            wins[away] += 1

        game_index += 1

    return high_seed_team if wins[high_seed_team] > wins[low_seed_team] else low_seed_team


def _conference_seeds(
    wins: Dict[str, int],
    conference_name: str,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Return top 8 teams and seed map (team -> 1..8) for a conference.
    """
    teams_conf = [
        t for t, conf in TEAM_CONFERENCE.items() if conf == conference_name and t in wins
    ]
    ranked = sorted(teams_conf, key=lambda t: (-wins[t], t))  # by wins desc, then name
    top8 = ranked[:8]
    seed_map = {team: i + 1 for i, team in enumerate(top8)}
    return top8, seed_map


def _simulate_series_with_seeds(
    team_a: str,
    team_b: str,
    seed_map: Dict[str, int],
    ratings: Dict[str, float],
) -> str:
    """
    Wrapper: determines which team has home court based on seed number.
    Lower seed number = better seed.
    """
    seed_a = seed_map[team_a]
    seed_b = seed_map[team_b]

    if seed_a < seed_b:
        high, low = team_a, team_b
    else:
        high, low = team_b, team_a

    return simulate_series(high, low, ratings)


def _simulate_conference_playoffs(
    seeds: List[str],
    seed_map: Dict[str, int],
    ratings: Dict[str, float],
) -> str:
    """
    Simulate conference bracket (no play-in, classic 1–8 format).
    seeds: [seed1, seed2, ..., seed8]
    """
    s1, s2, s3, s4, s5, s6, s7, s8 = seeds[:8]

    # Round 1
    w1 = _simulate_series_with_seeds(s1, s8, seed_map, ratings)  # 1 vs 8
    w2 = _simulate_series_with_seeds(s4, s5, seed_map, ratings)  # 4 vs 5
    w3 = _simulate_series_with_seeds(s3, s6, seed_map, ratings)  # 3 vs 6
    w4 = _simulate_series_with_seeds(s2, s7, seed_map, ratings)  # 2 vs 7

    # Round 2 (fixed bracket)
    sf1 = _simulate_series_with_seeds(w1, w2, seed_map, ratings)
    sf2 = _simulate_series_with_seeds(w4, w3, seed_map, ratings)

    # Conference Finals
    conf_champ = _simulate_series_with_seeds(sf1, sf2, seed_map, ratings)
    return conf_champ


def simulate_playoffs_once(wins: Dict[str, int]) -> str:
    """
    Given a set of regular-season wins, simulate full NBA playoffs and
    return the champion team name.
    """
    ratings = build_ratings(wins)

    east_seeds, east_seed_map = _conference_seeds(wins, "East")
    west_seeds, west_seed_map = _conference_seeds(wins, "West")

    if len(east_seeds) < 8 or len(west_seeds) < 8:
        raise ValueError("Not enough teams with wins to form 8 seeds per conference.")

    east_champ = _simulate_conference_playoffs(east_seeds, east_seed_map, ratings)
    west_champ = _simulate_conference_playoffs(west_seeds, west_seed_map, ratings)

    # Finals: home court to team with more wins
    if wins[east_champ] > wins[west_champ]:
        high, low = east_champ, west_champ
    else:
        high, low = west_champ, east_champ

    champion = simulate_series(high, low, ratings)
    return champion


def simulate_championships(
    schedule: pd.DataFrame,
    model,
    feature_cols: List[str],
    n_sims: int = 200,
) -> pd.DataFrame:
    """
    Run full (season + playoffs) simulations n_sims times and
    return a table of championship odds.
    """
    counts = Counter()

    for _ in range(n_sims):
        wins = simulate_one_regular_season(schedule, model, feature_cols)
        champ = simulate_playoffs_once(wins)
        counts[champ] += 1

    total = n_sims
    rows = []
    for team, c in counts.items():
        prob = c / total
        rows.append(
            {
                "team": team,
                "titles": c,
                "title_prob": prob,
                "title_prob_pct": round(prob * 100, 2),
            }
        )

    df = pd.DataFrame(rows).sort_values("title_prob", ascending=False)
    return df
