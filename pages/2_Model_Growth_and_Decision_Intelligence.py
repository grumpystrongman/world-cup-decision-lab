from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Growth", layout="wide")
st.title("Model Growth + Decision Intelligence")
st.caption("How the project evolved from a prediction model into a decision-support platform.")

REALITY_PATH = Path("data/actual/model_reality_check.csv")
MODEL_PATH = Path("data/processed/match_model.joblib")

st.header("The core idea")
st.markdown(
    """
Elo is a strong backbone for head-to-head probabilities, but tournament football does not behave like a smooth rating curve.

Form spikes, tactical matchups, group incentives, travel fatigue, and knockout psychology create discontinuities. Ten thousand simulations can smooth uncertainty, but they cannot magically predict what the model never measured.

This platform exists to measure those assumptions, expose their impact, and show how outcomes change when reality changes.
"""
)

st.header("Model evolution")
model_steps = pd.DataFrame(
    [
        {
            "stage": "1. Elo backbone",
            "what_changed": "Team strength was estimated from historical international results.",
            "why_it_mattered": "Created a transparent, explainable baseline instead of a black-box guess.",
        },
        {
            "stage": "2. Recent form",
            "what_changed": "Added recent points and goal-difference features.",
            "why_it_mattered": "Captured teams that were improving or fading relative to their long-term rating.",
        },
        {
            "stage": "3. Draw/upset risk",
            "what_changed": "Added close-match, uncertainty, draw-likelihood, and upset-risk features.",
            "why_it_mattered": "Reduced the model's tendency to treat favorites as inevitable winners.",
        },
        {
            "stage": "4. Probability calibration",
            "what_changed": "Added calibrated probabilities and conservative smoothing.",
            "why_it_mattered": "Made the model less confidently wrong, improving decision-quality probabilities.",
        },
        {
            "stage": "5. Draw-aware two-stage model",
            "what_changed": "Separated draw detection from decisive win prediction.",
            "why_it_mattered": "Tournament football has a draw problem; this makes the model explicitly reason about it.",
        },
        {
            "stage": "6. Discontinuity features",
            "what_changed": "Added form spike, form crash, favorite fragility, tactical mismatch, and pressure proxies.",
            "why_it_mattered": "Models the places where tournament football breaks from a smooth rating curve.",
        },
        {
            "stage": "7. Scenario + blast radius layer",
            "what_changed": "Added player-loss, fatigue, tactical disadvantage, and team-shock scenarios.",
            "why_it_mattered": "Shows who gains, who loses, and how the tournament changes when assumptions change.",
        },
    ]
)
st.dataframe(model_steps, use_container_width=True)

st.header("Live validation")
if REALITY_PATH.exists():
    scored = pd.read_csv(REALITY_PATH)
    if not scored.empty:
        scored["correct"] = scored["correct"].astype(bool)
        matches = len(scored)
        accuracy = scored["correct"].mean()
        avg_conf = scored["model_confidence"].mean()
        draw_rows = scored[scored["actual_outcome"] == "draw"]
        draw_accuracy = draw_rows["correct"].mean() if not draw_rows.empty else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches scored", f"{matches:,}")
        c2.metric("Live accuracy", f"{accuracy:.1%}")
        c3.metric("Average confidence", f"{avg_conf:.1%}")
        c4.metric("Draw accuracy", f"{draw_accuracy:.1%}")

        by_outcome = (
            scored.groupby("actual_outcome", as_index=False)
            .agg(matches=("match_id", "count"), accuracy=("correct", "mean"), avg_confidence=("model_confidence", "mean"))
            .sort_values("matches", ascending=False)
        )
        st.plotly_chart(px.bar(by_outcome, x="actual_outcome", y="accuracy", title="Reality Check: accuracy by actual outcome"), use_container_width=True)
    else:
        st.info("Reality-check file exists, but it does not contain scored matches yet.")
else:
    st.info("Run `python scripts/fetch_actual_results.py` and `python scripts/score_actual_results.py` to populate live validation.")

st.header("Blast radius impact analysis")
st.markdown(
    """
A normal prediction model asks: **Who is most likely to win?**

This platform asks a better question: **What changes when an assumption changes?**

The Scenario Lab runs a baseline tournament and a scenario tournament, then compares every team's probability movement. That creates a blast-radius view:

- Which team was directly harmed?
- Which teams benefited indirectly?
- Which teams lost opportunity?
- How many teams moved materially?
- How sensitive was the tournament to one assumption?

That matters because the most important effect is often not the obvious one. If France weakens, the question is not only how much France falls. The decision-support question is who benefits, who now has a better path, and how the bracket pressure moves.
"""
)

blast = pd.DataFrame(
    [
        {"component": "Baseline simulation", "purpose": "Establish the expected tournament state."},
        {"component": "Scenario simulation", "purpose": "Inject an assumption: player loss, fatigue, tactical disadvantage, or team shock."},
        {"component": "Delta engine", "purpose": "Measure before/after probability movement for every team."},
        {"component": "Winners/losers", "purpose": "Identify indirect beneficiaries and collateral damage."},
        {"component": "Sensitivity index", "purpose": "Quantify how disruptive the scenario was."},
        {"component": "Executive brief", "purpose": "Translate the math into a decision-ready narrative."},
    ]
)
st.dataframe(blast, use_container_width=True)

st.header("Why this is decision support, not just prediction")
st.markdown(
    """
Prediction is a point of view about the future.

Decision intelligence is a system for interrogating the future.

This project is valuable because it does not stop at a single answer. It lets a leader test assumptions, inspect uncertainty, compare scenarios, and understand second-order effects. The World Cup is just the demonstration environment. The same architecture maps cleanly to healthcare operations, staffing, patient flow, supply chain disruption, financial forecasting, and enterprise risk.

The real product is not soccer prediction. The product is disciplined uncertainty management.
"""
)

architecture = pd.DataFrame(
    [
        ["Data layer", "Historical match results, live actual results, venue/travel inputs, manual scenario assumptions"],
        ["Feature layer", "Elo, recent form, volatility, draw risk, tactical proxies, pressure proxies, fatigue"],
        ["Model layer", "Draw-aware two-stage classifier plus calibrated probability smoothing"],
        ["Simulation layer", "Monte Carlo tournament engine and scenario reruns"],
        ["Decision layer", "Blast radius, sensitivity index, winners/losers, executive briefings"],
        ["Validation layer", "Reality Check scoring against actual results"],
    ],
    columns=["Layer", "Role"],
)
st.dataframe(architecture, use_container_width=True)

st.success(
    "This is the bees-knees version: it predicts, explains, stress-tests, validates, and learns where it is wrong."
)
