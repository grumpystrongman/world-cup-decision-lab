from pathlib import Path
import time

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ingest.load_data import load_manual_teams
from src.models.predict import make_match_row, predict_match
from src.simulation.tournament import run_tournament_simulation
from src.explainability.explain import (
    feature_importance_from_model,
    plain_english_match_explanation,
)

st.set_page_config(page_title="World Cup Decision Intelligence Lab", layout="wide")

MODEL_PATH = Path("data/processed/match_model.joblib")
RATINGS_PATH = Path("data/processed/team_ratings.csv")
PLAYER_SCENARIO_PATH = Path("data/manual/player_impact_scenarios.csv")

st.title("World Cup 2026 Decision Intelligence Lab")
st.caption("Prediction is table stakes. Decision intelligence is the product.")

if not MODEL_PATH.exists() or not RATINGS_PATH.exists():
    st.error("Run `python build_pipeline.py` first.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["pipeline"]
metrics = artifact["metrics"]

ratings = pd.read_csv(RATINGS_PATH)
teams = load_manual_teams()
team_list = sorted(ratings["team"].unique())

if PLAYER_SCENARIO_PATH.exists():
    player_scenarios = pd.read_csv(PLAYER_SCENARIO_PATH, sep="|")
else:
    player_scenarios = pd.DataFrame(columns=["team", "player", "role", "estimated_elo_impact", "reason"])


def run_with_timer(label, sims, adjustments):
    progress = st.progress(0)
    status = st.empty()
    start = time.perf_counter()

    def update_progress(value):
        progress.progress(min(1.0, float(value)))
        status.write(f"{label}: {int(float(value) * 100)}% complete")

    result = run_tournament_simulation(
        model,
        teams,
        ratings,
        n_sims=sims,
        adjustments=adjustments,
        progress_callback=update_progress,
    )

    elapsed = time.perf_counter() - start
    progress.progress(1.0)
    status.success(f"{label}: completed {sims:,} simulations in {elapsed:.2f} seconds.")
    return result, elapsed


def compare_runs(baseline, scenario):
    base = baseline[["team", "champion", "final", "semifinal", "quarterfinal", "round_of_16", "round_of_32"]].copy()
    scen = scenario[["team", "champion", "final", "semifinal", "quarterfinal", "round_of_16", "round_of_32"]].copy()

    merged = base.merge(scen, on="team", suffixes=("_baseline", "_scenario"))
    merged["champion_delta"] = merged["champion_scenario"] - merged["champion_baseline"]
    merged["final_delta"] = merged["final_scenario"] - merged["final_baseline"]
    merged["semifinal_delta"] = merged["semifinal_scenario"] - merged["semifinal_baseline"]
    merged["abs_champion_delta"] = merged["champion_delta"].abs()

    return merged.sort_values("abs_champion_delta", ascending=False).reset_index(drop=True)


def sensitivity_index(delta_df, target_team=None):
    total_movement = float(delta_df["champion_delta"].abs().sum())
    affected_teams = int((delta_df["champion_delta"].abs() >= 0.0025).sum())
    target_hit = 0.0

    if target_team and target_team in set(delta_df["team"]):
        target_hit = float(abs(delta_df.loc[delta_df["team"] == target_team, "champion_delta"].iloc[0]))

    raw_score = (total_movement * 150) + (affected_teams * 2) + (target_hit * 250)
    return int(max(0, min(100, round(raw_score))))


def scenario_summary_text(label, target_team, impact, baseline, scenario, delta_df, sensitivity, sims, elapsed):
    leader = scenario.iloc[0]
    biggest_winner = delta_df.sort_values("champion_delta", ascending=False).iloc[0]
    biggest_loser = delta_df.sort_values("champion_delta", ascending=True).iloc[0]

    if target_team in set(delta_df["team"]):
        target_row = delta_df[delta_df["team"] == target_team].iloc[0]
        before = target_row["champion_baseline"]
        after = target_row["champion_scenario"]
        change = target_row["champion_delta"]
    else:
        before = after = change = 0

    return f"""
### Executive Scenario Brief

**Scenario:** {label}

**Target-team impact:** {target_team} moved from **{before:.1%}** title probability to **{after:.1%}**, a change of **{change:+.1%}**.

**Current scenario leader:** {leader["team"]} at **{leader["champion"]:.1%}** championship probability.

**Biggest beneficiary:** {biggest_winner["team"]} **{biggest_winner["champion_delta"]:+.1%}**

**Biggest loser:** {biggest_loser["team"]} **{biggest_loser["champion_delta"]:+.1%}**

**Scenario Sensitivity Index:** **{sensitivity}/100**

**Model run:** {sims:,} baseline simulations + {sims:,} scenario simulations completed in **{elapsed:.2f} seconds**.

This is the difference between prediction and decision intelligence: we are not only asking who is favored. We are measuring how the tournament changes when assumptions change.
"""


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Command Center",
        "Scenario Lab",
        "Why This Beats Simple Prediction",
        "Executive Briefing",
    ]
)

with tab1:
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

        st.plotly_chart(
            px.bar(prob_df, x="outcome", y="probability", title="Match outcome probability"),
            use_container_width=True,
        )
        st.info(plain_english_match_explanation(home, away, probabilities, match_row))

    st.divider()
    st.subheader("Explainability")
    importance = feature_importance_from_model(model)
    st.dataframe(importance, use_container_width=True)

    st.plotly_chart(
        px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Model feature importance",
        ),
        use_container_width=True,
    )

    with st.expander("Model metrics"):
        st.json(metrics)

with tab2:
    st.subheader("Scenario Lab + Blast Radius Analysis")

    if teams.empty:
        st.warning("Add `data/manual/world_cup_2026_teams.csv` to run tournament simulations.")
        st.stop()

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        sims = st.slider("Simulations per run", 100, 100000, 5000, 100)

    with c2:
        scenario_type = st.selectbox(
            "Scenario type",
            [
                "Manual team-strength shock",
                "Key player unavailable",
                "Fatigue / travel burden",
                "Tactical disadvantage",
            ],
        )

    adjustments = {}
    target_team = None
    impact = 0
    scenario_label = "baseline"

    with c3:
        if scenario_type == "Manual team-strength shock":
            target_team = st.selectbox("Team to stress test", sorted(teams["team"].unique()))
            impact = st.slider("Elo impact", -250, 250, -75, 10)
            adjustments = {target_team: impact}
            scenario_label = f"{target_team} {impact:+d} Elo shock"

        elif scenario_type == "Key player unavailable":
            if player_scenarios.empty:
                st.error("Missing data/manual/player_impact_scenarios.csv")
            else:
                labels = (
                    player_scenarios["team"]
                    + " — "
                    + player_scenarios["player"]
                    + " ("
                    + player_scenarios["role"]
                    + ")"
                ).tolist()

                selected = st.selectbox("Player loss scenario", labels)
                selected_row = player_scenarios.iloc[labels.index(selected)]

                target_team = selected_row["team"]
                impact = int(selected_row["estimated_elo_impact"])
                adjustments = {target_team: impact}
                scenario_label = f"{target_team} without {selected_row['player']} ({impact:+d} Elo)"
                st.info(selected_row["reason"])

        elif scenario_type == "Fatigue / travel burden":
            target_team = st.selectbox("Team affected", sorted(teams["team"].unique()))
            severity = st.selectbox("Severity", ["Low", "Medium", "High"])
            impact_map = {"Low": -25, "Medium": -50, "High": -75}
            impact = impact_map[severity]
            adjustments = {target_team: impact}
            scenario_label = f"{target_team} {severity.lower()} fatigue / travel burden ({impact:+d} Elo)"

        elif scenario_type == "Tactical disadvantage":
            target_team = st.selectbox("Team disadvantaged", sorted(teams["team"].unique()))
            severity = st.selectbox("Severity", ["Low", "Medium", "High"])
            impact_map = {"Low": -20, "Medium": -45, "High": -70}
            impact = impact_map[severity]
            adjustments = {target_team: impact}
            scenario_label = f"{target_team} tactical disadvantage ({impact:+d} Elo)"

    st.write(f"### Scenario: {scenario_label}")

    if st.button("Run Blast Radius Analysis", type="primary"):
        st.session_state["scenario_label"] = scenario_label
        st.session_state["target_team"] = target_team
        st.session_state["impact"] = impact

        st.write("Running baseline and scenario simulations...")
        baseline, baseline_seconds = run_with_timer("Baseline", sims, {})
        scenario, scenario_seconds = run_with_timer("Scenario", sims, adjustments)

        delta_df = compare_runs(baseline, scenario)
        sensitivity = sensitivity_index(delta_df, target_team=target_team)

        st.session_state["baseline"] = baseline
        st.session_state["scenario"] = scenario
        st.session_state["delta_df"] = delta_df
        st.session_state["sensitivity"] = sensitivity
        st.session_state["elapsed"] = baseline_seconds + scenario_seconds
        st.session_state["sims"] = sims

    if "delta_df" in st.session_state:
        delta_df = st.session_state["delta_df"]
        scenario = st.session_state["scenario"]
        baseline = st.session_state["baseline"]
        sensitivity = st.session_state["sensitivity"]
        label = st.session_state["scenario_label"]
        target = st.session_state["target_team"]
        elapsed = st.session_state["elapsed"]
        sims = st.session_state["sims"]

        target_row = delta_df[delta_df["team"] == target].iloc[0] if target in set(delta_df["team"]) else None

        k1, k2, k3, k4 = st.columns(4)

        if target_row is not None:
            k1.metric(
                f"{target} title odds",
                f"{target_row['champion_scenario']:.1%}",
                f"{target_row['champion_delta']:+.1%}",
            )
        else:
            k1.metric("Target title odds", "N/A")

        k2.metric("Scenario Sensitivity Index", f"{sensitivity}/100")
        k3.metric("Teams materially affected", int((delta_df["champion_delta"].abs() >= 0.0025).sum()))
        k4.metric("Total runtime", f"{elapsed:.2f}s")

        st.markdown(scenario_summary_text(label, target, impact, baseline, scenario, delta_df, sensitivity, sims, elapsed))

        st.subheader("Blast Radius: Biggest Winners and Losers")

        winners = delta_df.sort_values("champion_delta", ascending=False).head(8)
        losers = delta_df.sort_values("champion_delta", ascending=True).head(8)

        left, right = st.columns(2)

        with left:
            st.write("#### Biggest Winners")
            st.dataframe(
                winners[["team", "champion_baseline", "champion_scenario", "champion_delta"]],
                use_container_width=True,
            )

            st.plotly_chart(
                px.bar(
                    winners.sort_values("champion_delta"),
                    x="champion_delta",
                    y="team",
                    orientation="h",
                    title="Teams gaining title probability",
                ),
                use_container_width=True,
            )

        with right:
            st.write("#### Biggest Losers")
            st.dataframe(
                losers[["team", "champion_baseline", "champion_scenario", "champion_delta"]],
                use_container_width=True,
            )

            st.plotly_chart(
                px.bar(
                    losers.sort_values("champion_delta", ascending=False),
                    x="champion_delta",
                    y="team",
                    orientation="h",
                    title="Teams losing title probability",
                ),
                use_container_width=True,
            )

        st.subheader("Before vs After Championship Probability")

        top_compare = (
            delta_df.sort_values("champion_scenario", ascending=False)
            .head(16)
            [["team", "champion_baseline", "champion_scenario"]]
            .melt(id_vars="team", var_name="run", value_name="champion_probability")
        )

        fig = px.bar(
            top_compare,
            x="champion_probability",
            y="team",
            color="run",
            barmode="group",
            orientation="h",
            title="Baseline vs Scenario",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Waterfall: Target Team Probability Movement")

        if target_row is not None:
            before = float(target_row["champion_baseline"])
            after = float(target_row["champion_scenario"])
            change = after - before

            waterfall = go.Figure(
                go.Waterfall(
                    name="Probability movement",
                    orientation="v",
                    measure=["absolute", "relative", "total"],
                    x=["Baseline", "Scenario impact", "After scenario"],
                    y=[before, change, after],
                    text=[f"{before:.1%}", f"{change:+.1%}", f"{after:.1%}"],
                    textposition="outside",
                )
            )
            waterfall.update_layout(title=f"{target} Championship Probability Movement", yaxis_tickformat=".0%")
            st.plotly_chart(waterfall, use_container_width=True)

with tab3:
    st.subheader("Why this is vastly superior to simple prediction")

    st.markdown(
        """
A simple prediction model answers:

**Who is most likely to win?**

That is useful, but shallow.

This platform answers better questions:

- Why did the probability change?
- What happens when a key assumption changes?
- Who benefits from another team's weakness?
- How many teams are affected by one scenario?
- How sensitive is the tournament to one player, one fatigue burden, or one tactical mismatch?

### Prediction vs Decision Intelligence

**Simple prediction:**

France has an X% chance to win.

**Decision intelligence:**

France has an X% chance to win. If France loses a system-critical player, its odds fall by Y, England gains Z, Brazil gains A, and the scenario creates a measurable blast radius across B teams.

That is the difference.

Prediction gives an answer.

Decision intelligence explains the consequences.
"""
    )

    st.write("### Backend architecture")

    arch = pd.DataFrame(
        [
            ["Data", "49,413 historical international matches"],
            ["Ratings", "Elo-style team strength engine"],
            ["Model", "Machine-learning match outcome classifier"],
            ["Simulation", "Monte Carlo tournament engine"],
            ["Scenario Layer", "Team shocks, player loss, fatigue, tactical disadvantage"],
            ["Explainability", "Feature importance, probability deltas, blast radius"],
            ["Executive Layer", "Plain-English scenario briefings"],
        ],
        columns=["Layer", "What it does"],
    )

    st.dataframe(arch, use_container_width=True)

with tab4:
    st.subheader("Executive Briefing")

    if "delta_df" not in st.session_state:
        st.warning("Run a Blast Radius Analysis first. Then come back here for the screenshot.")
    else:
        delta_df = st.session_state["delta_df"]
        scenario = st.session_state["scenario"]
        label = st.session_state["scenario_label"]
        target = st.session_state["target_team"]
        sensitivity = st.session_state["sensitivity"]
        sims = st.session_state["sims"]
        elapsed = st.session_state["elapsed"]

        target_row = delta_df[delta_df["team"] == target].iloc[0] if target in set(delta_df["team"]) else None
        leader = scenario.iloc[0]
        winners = delta_df.sort_values("champion_delta", ascending=False).head(3)
        losers = delta_df.sort_values("champion_delta", ascending=True).head(3)

        st.markdown(
            f"""
# World Cup 2026 Decision Intelligence Lab

## What happens when assumptions change?

### Scenario
**{label}**

### Model Foundation
**49,413 historical matches**  
**336 national teams rated**  
**{sims:,} baseline simulations + {sims:,} scenario simulations**  
**Completed in {elapsed:.2f} seconds**

---
"""
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            if target_row is not None:
                st.metric(
                    f"{target} champion odds",
                    f"{target_row['champion_scenario']:.1%}",
                    f"{target_row['champion_delta']:+.1%}",
                )

        with c2:
            st.metric("Scenario Sensitivity Index", f"{sensitivity}/100")

        with c3:
            st.metric("Scenario leader", leader["team"], f"{leader['champion']:.1%}")

        left, right = st.columns(2)

        with left:
            st.write("### Biggest Winners")
            for _, row in winners.iterrows():
                st.write(f"**{row['team']}** {row['champion_delta']:+.1%}")

        with right:
            st.write("### Biggest Losers")
            for _, row in losers.iterrows():
                st.write(f"**{row['team']}** {row['champion_delta']:+.1%}")

        st.info(
            "Prediction tells you what might happen. Decision intelligence shows how outcomes change when assumptions change."
        )
