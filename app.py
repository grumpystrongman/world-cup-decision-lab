from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ingest.load_data import load_manual_teams
from src.models.predict import make_match_row, predict_match
from src.simulation.tournament import run_tournament_simulation
from src.explainability.explain import (
    executive_tournament_summary,
    feature_importance_from_model,
    plain_english_match_explanation,
)

st.set_page_config(page_title="World Cup Decision Intelligence Lab", layout="wide")
st.title("World Cup 2026 Decision Intelligence Lab")
st.caption("Prediction is table stakes. Scenario intelligence is the product.")

MODEL_PATH = Path("data/processed/match_model.joblib")
RATINGS_PATH = Path("data/processed/team_ratings.csv")

if not MODEL_PATH.exists() or not RATINGS_PATH.exists():
    st.warning("Run `python build_pipeline.py --use-sample` for demo data or add data/raw/results.csv and run `python build_pipeline.py`.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["pipeline"]
metrics = artifact["metrics"]
ratings = pd.read_csv(RATINGS_PATH)
teams = load_manual_teams()
team_list = sorted(ratings["team"].unique())

left, right = st.columns([1, 1])

with left:
    st.subheader("Team Strength Engine")
    top_n = st.slider("Show top teams", 5, min(40, len(ratings)), min(15, len(ratings)))
    st.dataframe(ratings.head(top_n), use_container_width=True)
    fig = px.bar(
        ratings.head(top_n).sort_values("elo"),
        x="elo",
        y="team",
        orientation="h",
        title="Current Elo-style team strength",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Match Predictor")
    default_home = "France" if "France" in team_list else team_list[0]
    default_away = "England" if "England" in team_list else team_list[min(1, len(team_list) - 1)]
    home = st.selectbox("Team 1", team_list, index=team_list.index(default_home))
    away = st.selectbox("Team 2", team_list, index=team_list.index(default_away))
    home_adjustment = st.slider(f"{home} scenario adjustment", -250, 250, 0, 10)
    away_adjustment = st.slider(f"{away} scenario adjustment", -250, 250, 0, 10)
    match_row = make_match_row(
        home,
        away,
        ratings,
        neutral=True,
        team_adjustments={home: home_adjustment, away: away_adjustment},
    )
    probabilities = predict_match(model, match_row)
    prob_df = pd.DataFrame({"outcome": list(probabilities.keys()), "probability": list(probabilities.values())})
    st.plotly_chart(px.bar(prob_df, x="outcome", y="probability", title="Match outcome probability"), use_container_width=True)
    st.info(plain_english_match_explanation(home, away, probabilities, match_row))

st.divider()
st.subheader("Scenario Lab + Tournament Simulator")

if teams.empty:
    st.warning("Add `data/manual/world_cup_2026_teams.csv` to run tournament simulations.")
else:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sims = st.slider("Number of simulations", 100, 25000, 5000, 100)
    with c2:
        scenario_team = st.selectbox("Team to stress test", ["None"] + sorted(teams["team"].unique()))
    with c3:
        scenario_delta = st.slider("Elo scenario impact", -250, 250, 0, 10)

    adjustments = {} if scenario_team == "None" else {scenario_team: scenario_delta}
    scenario_label = "baseline" if scenario_team == "None" else f"{scenario_team} {scenario_delta:+d} Elo"

    if st.button("Run tournament simulation", type="primary"):
        sim = run_tournament_simulation(model, teams, ratings, n_sims=sims, adjustments=adjustments)
        st.success(executive_tournament_summary(sim, scenario_label=scenario_label))
        st.dataframe(sim, use_container_width=True)
        st.plotly_chart(
            px.bar(sim.head(16).sort_values("champion"), x="champion", y="team", orientation="h", title="Simulated title probability"),
            use_container_width=True,
        )

st.divider()
st.subheader("Explainability")
importance = feature_importance_from_model(model)
st.dataframe(importance, use_container_width=True)
st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", title="Model feature importance"), use_container_width=True)

with st.expander("Model metrics"):
    st.json(metrics)
