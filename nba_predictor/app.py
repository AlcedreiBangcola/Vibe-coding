import streamlit as st
import pandas as pd
from joblib import load

from src.build_dataset import prepare_games, build_team_features, make_model_dataset
from src.simulate_season import simulate_regular_season
from src.simulate_playoffs import simulate_championships


# ---------- DATA & MODEL LOADING ----------

@st.cache_data
def load_data():
    # Uses your historical data file from the CLI pipeline
    games = prepare_games("data/raw/games_2015_2024.csv")
    team_games = build_team_features(games)
    dataset = make_model_dataset(games, team_games)
    return games, dataset


@st.cache_data
def compute_baseline_stats(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Compute baseline team strength from the most recent historical season.
    We’ll use this to build features for future or hypothetical schedules.
    """
    last_season = dataset["Season"].max()
    df = dataset[dataset["Season"] == last_season]

    # Use average home rolling stats as team baselines
    home_stats = (
        df.groupby("home_team")[
            ["home_rolling_pd", "home_rolling_pf", "home_rolling_pa"]
        ]
        .mean()
        .rename(
            columns={
                "home_rolling_pd": "baseline_pd",
                "home_rolling_pf": "baseline_pf",
                "home_rolling_pa": "baseline_pa",
            }
        )
        .reset_index()
        .rename(columns={"home_team": "team"})
    )
    return home_stats


@st.cache_resource
def load_model():
    model = load("models/game_model.joblib")
    feature_cols = load("models/feature_cols.joblib")
    return model, feature_cols

@st.cache_data
def load_team_roster_strength() -> pd.DataFrame | None:
    """
    Load per-team roster strength ratings from CSV, if it exists.
    CSV path: data/processed/team_roster_strength.csv
    Must have columns: team, roster_strength
    """
    try:
        df = pd.read_csv("data/processed/team_roster_strength.csv")
        if "team" not in df.columns or "roster_strength" not in df.columns:
            raise ValueError("team_roster_strength.csv must have 'team' and 'roster_strength' columns.")
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def compute_baseline_stats(dataset: pd.DataFrame, roster_strength: pd.DataFrame | None) -> pd.DataFrame:
    """
    Compute baseline team strength from the most recent historical season.
    Optionally adjust these based on a roster_strength table.
    """
    last_season = dataset["Season"].max()
    df = dataset[dataset["Season"] == last_season]

    # Average home rolling stats as team baselines
    home_stats = (
        df.groupby("home_team")[
            ["home_rolling_pd", "home_rolling_pf", "home_rolling_pa"]
        ]
        .mean()
        .rename(
            columns={
                "home_rolling_pd": "baseline_pd",
                "home_rolling_pf": "baseline_pf",
                "home_rolling_pa": "baseline_pa",
            }
        )
        .reset_index()
        .rename(columns={"home_team": "team"})
    )

    # If we have roster_strength, merge it and use it to tweak baselines
    if roster_strength is not None:
        merged = home_stats.merge(roster_strength, on="team", how="left")

        # Fill missing roster_strength with league average
        mean_rs = merged["roster_strength"].mean()
        merged["roster_strength"] = merged["roster_strength"].fillna(mean_rs)

        # Center roster_strength around 0
        merged["rs_centered"] = merged["roster_strength"] - mean_rs

        # Scale factor: how strongly roster strength impacts baseline stats
        alpha_pd = 0.15
        alpha_pf = 0.10
        alpha_pa = 0.10

        merged["baseline_pd"] = merged["baseline_pd"] + alpha_pd * merged["rs_centered"]
        merged["baseline_pf"] = merged["baseline_pf"] + alpha_pf * merged["rs_centered"]
        merged["baseline_pa"] = merged["baseline_pa"] - alpha_pa * merged["rs_centered"]

        return merged[["team", "baseline_pd", "baseline_pf", "baseline_pa"]]

    # No roster info, just return baselines from data
    return home_stats

# ---------- FEATURE BUILDING FOR CUSTOM SCHEDULES ----------

def build_features_for_custom_schedule(
    schedule_df: pd.DataFrame,
    baseline_stats: pd.DataFrame,
    season_label: int,
) -> pd.DataFrame:
    """
    Given a schedule with 'home_team' and 'away_team', attach baseline
    team strengths and compute pd_diff, pf_diff, pa_diff, plus neutral
    rest_diff and b2b_diff for use with the trained model.

    For future/hypothetical seasons we don't have real rest info, so
    we set rest_diff and b2b_diff to 0 (no advantage).
    """
    sched = schedule_df.copy()

    if "home_team" not in sched.columns or "away_team" not in sched.columns:
        raise ValueError("Schedule must have 'home_team' and 'away_team' columns.")

    sched["Season"] = season_label

    # Build separate home + away baseline tables to avoid suffix confusion
    home_baseline = baseline_stats.rename(
        columns={
            "team": "home_team",
            "baseline_pd": "home_rolling_pd",
            "baseline_pf": "home_rolling_pf",
            "baseline_pa": "home_rolling_pa",
        }
    )

    away_baseline = baseline_stats.rename(
        columns={
            "team": "away_team",
            "baseline_pd": "away_rolling_pd",
            "baseline_pf": "away_rolling_pf",
            "baseline_pa": "away_rolling_pa",
        }
    )

    # Merge baselines
    sched = sched.merge(home_baseline, on="home_team", how="left")
    sched = sched.merge(away_baseline, on="away_team", how="left")

    # Fill missing baseline values with 0 (league-average-ish)
    cols_to_fill = [
        "home_rolling_pd",
        "home_rolling_pf",
        "home_rolling_pa",
        "away_rolling_pd",
        "away_rolling_pf",
        "away_rolling_pa",
    ]
    for c in cols_to_fill:
        if c not in sched.columns:
            sched[c] = 0.0
        else:
            sched[c] = sched[c].fillna(0.0)

    # Core feature diffs
    sched["pd_diff"] = sched["home_rolling_pd"] - sched["away_rolling_pd"]
    sched["pf_diff"] = sched["home_rolling_pf"] - sched["away_rolling_pf"]
    sched["pa_diff"] = sched["home_rolling_pa"] - sched["away_rolling_pa"]

    # NEW: neutral rest / back-to-back differences for future schedules
    # (the trained model expects these columns to exist)
    sched["rest_diff"] = 0.0
    sched["b2b_diff"] = 0.0

    return sched

def fetch_bref_schedule(season: int) -> pd.DataFrame:
    """
    Fetch the *full league schedule* for a season from Basketball-Reference.

    season: year the season ends (e.g. 2025 for 2024–25).
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    df_list = pd.read_html(url)

    games = df_list[0]
    games = games[games["Date"] != "Playoffs"]  # drop playoffs rows if any

    games = games.rename(
        columns={
            "Visitor/Neutral": "away_team",
            "Home/Neutral": "home_team",
        }
    )

    # We only need team matchups + date for simulation
    schedule = games[["Date", "home_team", "away_team"]].copy()
    schedule["Season"] = season
    return schedule


# ---------- STREAMLIT APP ----------

def run_full_simulation(
    schedule_features: pd.DataFrame,
    model,
    feature_cols,
    n_sims: int,
    label: str,
):
    """Helper to run regular season + playoffs and display tables."""
    with st.spinner(f"Simulating regular season for {label}..."):
        season_summary = simulate_regular_season(
            schedule_features, model, feature_cols, n_sims=n_sims
        )

    st.subheader(f"Projected Regular-Season Wins ({label})")
    st.dataframe(season_summary.reset_index(drop=True), use_container_width=True)

    with st.spinner(f"Simulating playoffs and championships for {label}..."):
        title_odds = simulate_championships(
            schedule_features, model, feature_cols, n_sims=n_sims
        )

    st.subheader(f"Championship Odds ({label})")
    st.dataframe(title_odds.reset_index(drop=True), use_container_width=True)

    if not title_odds.empty:
        top_row = title_odds.iloc[0]
        st.markdown(
            f"### 🔮 Model’s top pick: **{top_row['team']}** "
            f"({top_row['title_prob_pct']}% title chance in this simulation)"
        )


def main():
    st.title("🏀 NBA Season & Playoff Predictor")

    st.write(
        """
        This app:
        - Learns a game model from past seasons (2015–2024)
        - Simulates full regular seasons and playoffs
        - Can use **real past seasons**, the **current/future schedule** (e.g. 2025),
          or **your own hypothetical schedule**.
        """
    )

    games, dataset = load_data()
    model, feature_cols = load_model()
    roster_strength = load_team_roster_strength()
    baseline_stats = compute_baseline_stats(dataset, roster_strength)

    # Debug / verification: show that roster_strength is being used
    if roster_strength is not None:
        with st.expander("🔍 Show team roster strengths (from player stats)"):
            st.write(
                "These values are computed from player advanced stats on Basketball-Reference "
                "(e.g. sum of top players' VORP × minutes per team). Higher = stronger roster."
            )
            st.dataframe(
                roster_strength.sort_values("roster_strength", ascending=False),
                use_container_width=True,
            )

        debug_baseline = baseline_stats.merge(roster_strength, on="team", how="left")
        with st.expander("🔍 Show baseline team stats used by the model"):
            st.write(
                "These are the baseline per-team stats (point differential, points for/against) that "
                "feed into the game model for future seasons. They already include adjustments based "
                "on the roster strengths above."
            )
            st.dataframe(debug_baseline, use_container_width=True)
    else:
        st.warning(
            "No team_roster_strength.csv found. The model is currently using only historical team "
            "performance, not player-based roster strength."
        )

    mode = st.radio(
        "Choose mode",
        [
            "Historical season (what-if re-run)",
            "Real current/future season (auto schedule, e.g. 2025)",
            "Custom hypothetical season (upload CSV)",
        ],
    )

    n_sims = st.slider(
        "Number of simulations",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        help="More simulations = smoother results but slower.",
    )

    # ---------- MODE 1: Historical season ----------

    if mode == "Historical season (what-if re-run)":
        seasons = sorted(games["Season"].unique())
        default_index = len(seasons) - 1  # most recent historical season in your data

        season = st.selectbox(
            "Choose historical season schedule to simulate", seasons, index=default_index
        )

        if st.button("Run historical season simulation"):
            schedule_features = dataset[dataset["Season"] == season].copy()
            run_full_simulation(
                schedule_features, model, feature_cols, n_sims, label=str(season)
            )

    # ---------- MODE 2: Real current/future season (e.g. 2025) ----------

    elif mode == "Real current/future season (auto schedule, e.g. 2025)":
        season_year = st.number_input(
            "Season ending year (e.g. 2025 for the 2024–25 season)",
            min_value=2025,
            max_value=2100,
            value=2025,
            step=1,
        )

        st.info(
            "This will pull the full league schedule from Basketball-Reference "
            "for that season, then simulate the entire year + playoffs."
        )

        if st.button("Run future season simulation"):
            try:
                schedule_df = fetch_bref_schedule(season_year)
                schedule_features = build_features_for_custom_schedule(
                    schedule_df, baseline_stats, season_label=season_year
                )
                run_full_simulation(
                    schedule_features,
                    model,
                    feature_cols,
                    n_sims,
                    label=f"{season_year} (auto schedule)",
                )
            except Exception as e:
                st.error(f"Error fetching or simulating schedule: {e}")

    # ---------- MODE 3: Custom hypothetical season ----------

    else:
        season_label = st.number_input(
            "Label for hypothetical season (e.g. 2030)",
            min_value=2025,
            max_value=2100,
            value=2030,
            step=1,
        )

        st.write(
            """
            Upload a CSV with at least these columns:
            - **home_team**
            - **away_team**
            
            Optional columns (ignored by the model but useful for you):
            - Date, game_id, etc.
            """
        )

        uploaded = st.file_uploader("Upload schedule CSV", type=["csv"])

        if uploaded is not None and st.button("Run simulation on custom schedule"):
            try:
                schedule_df = pd.read_csv(uploaded)
                schedule_features = build_features_for_custom_schedule(
                    schedule_df, baseline_stats, season_label=int(season_label)
                )
                run_full_simulation(
                    schedule_features,
                    model,
                    feature_cols,
                    n_sims,
                    label=f"Hypothetical {int(season_label)}",
                )
            except Exception as e:
                st.error(f"Error reading or simulating custom schedule: {e}")

    st.caption(
        "Note: This is a toy model using historical results and simple assumptions. "
        "Fun for exploration, not for serious betting 😄"
    )

if __name__ == "__main__":
    main()
