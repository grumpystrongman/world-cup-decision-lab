from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REALITY_PATH = Path("data/actual/model_reality_check.csv")
RESULTS_PATH = Path("data/actual/world_cup_results.csv")

st.set_page_config(page_title="Reality Check", layout="wide")
st.title("Reality Check")
st.caption("How the decision model is performing against actual tournament results.")

if not REALITY_PATH.exists():
    st.warning(
        "No scored actual results found yet. Run `python scripts/fetch_actual_results.py` and "
        "`python scripts/score_actual_results.py`, then refresh this page."
    )
    st.stop()

scored = pd.read_csv(REALITY_PATH)
if scored.empty:
    st.info("The reality-check file exists, but there are no completed matches to score yet.")
    st.stop()

scored["correct"] = scored["correct"].astype(bool)
scored["date"] = pd.to_datetime(scored["date"], errors="coerce")
scored["match"] = scored["home_team"] + " vs " + scored["away_team"]
scored["confidence_bucket"] = pd.cut(
    scored["model_confidence"],
    bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    labels=["<=40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"],
    include_lowest=True,
)

matches_scored = len(scored)
correct = int(scored["correct"].sum())
incorrect = matches_scored - correct
accuracy = scored["correct"].mean()
avg_confidence = scored["model_confidence"].mean()
draw_rows = scored[scored["actual_outcome"] == "draw"]
draw_accuracy = draw_rows["correct"].mean() if not draw_rows.empty else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Matches scored", f"{matches_scored:,}")
c2.metric("Accuracy", f"{accuracy:.1%}")
c3.metric("Correct", f"{correct:,}")
c4.metric("Missed", f"{incorrect:,}")
c5.metric("Average confidence", f"{avg_confidence:.1%}")

st.info(
    "Read this correctly: accuracy tells us how often the top pick was right; calibration tells us "
    "whether confidence was trustworthy. A model can have useful accuracy and still need calibration."
)

st.subheader("Model scorecard")
scorecard = pd.DataFrame(
    [
        {"metric": "Matches scored", "value": matches_scored},
        {"metric": "Correct picks", "value": correct},
        {"metric": "Missed picks", "value": incorrect},
        {"metric": "Accuracy", "value": round(accuracy, 3)},
        {"metric": "Average model confidence", "value": round(avg_confidence, 3)},
        {"metric": "Draw accuracy", "value": round(float(draw_accuracy), 3)},
    ]
)
st.dataframe(scorecard, use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("Correct vs missed")
    outcome_counts = pd.DataFrame(
        [
            {"result": "Correct", "count": correct},
            {"result": "Missed", "count": incorrect},
        ]
    )
    st.plotly_chart(
        px.bar(outcome_counts, x="result", y="count", title="Prediction outcomes"),
        use_container_width=True,
    )

with right:
    st.subheader("Accuracy by actual outcome")
    by_outcome = (
        scored.groupby("actual_outcome", as_index=False)
        .agg(matches=("match_id", "count"), accuracy=("correct", "mean"), avg_confidence=("model_confidence", "mean"))
        .sort_values("matches", ascending=False)
    )
    st.dataframe(by_outcome, use_container_width=True)
    st.plotly_chart(
        px.bar(by_outcome, x="actual_outcome", y="accuracy", title="Accuracy by actual result"),
        use_container_width=True,
    )

st.subheader("Biggest confident misses")
misses = scored[~scored["correct"]].sort_values("model_confidence", ascending=False).copy()
if misses.empty:
    st.success("No misses yet.")
else:
    st.dataframe(
        misses[
            [
                "date",
                "match",
                "home_score",
                "away_score",
                "actual_outcome",
                "model_pick",
                "model_confidence",
                "home_win_prob",
                "draw_prob",
                "away_win_prob",
            ]
        ],
        use_container_width=True,
    )
    st.plotly_chart(
        px.bar(
            misses.head(12).sort_values("model_confidence"),
            x="model_confidence",
            y="match",
            orientation="h",
            title="Most confident misses",
        ),
        use_container_width=True,
    )

st.subheader("Calibration sanity check")
calibration = (
    scored.groupby("confidence_bucket", observed=False)
    .agg(matches=("match_id", "count"), observed_accuracy=("correct", "mean"), avg_confidence=("model_confidence", "mean"))
    .reset_index()
)
st.dataframe(calibration, use_container_width=True)
st.plotly_chart(
    px.line(
        calibration,
        x="confidence_bucket",
        y=["observed_accuracy", "avg_confidence"],
        markers=True,
        title="Observed accuracy vs stated confidence",
    ),
    use_container_width=True,
)

st.subheader("All scored matches")
st.dataframe(
    scored[
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "actual_outcome",
            "model_pick",
            "model_confidence",
            "home_win_prob",
            "draw_prob",
            "away_win_prob",
            "correct",
        ]
    ].sort_values("date", ascending=False),
    use_container_width=True,
)

st.markdown(
    """
### What this tells us

The model is being evaluated against reality, not just trained history. The immediate lesson is whether the model is directionally useful and whether its confidence is trustworthy.

If accuracy holds but confident misses remain high, the next model improvement is probability calibration and better draw handling. That is why the training pipeline now uses calibrated probabilities rather than raw random forest probabilities.
"""
)
