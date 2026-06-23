from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.models.predict import make_match_row, predict_match

RESULTS_PATH = REPO_ROOT / "data/actual/world_cup_results.csv"
MODEL_PATH = REPO_ROOT / "data/processed/match_model.joblib"
RATINGS_PATH = REPO_ROOT / "data/processed/team_ratings.csv"
OUT_PATH = REPO_ROOT / "data/actual/model_reality_check.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ORDERED_LABELS = ["away_win", "draw", "home_win"]


def score_predictions() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Missing data/actual/world_cup_results.csv. Run scripts/fetch_actual_results.py first.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing data/processed/match_model.joblib. Run build_pipeline.py or pull model artifacts.")
    if not RATINGS_PATH.exists():
        raise FileNotFoundError("Missing data/processed/team_ratings.csv. Run build_pipeline.py or pull model artifacts.")

    results = pd.read_csv(RESULTS_PATH)
    artifact = joblib.load(MODEL_PATH)
    model = artifact["pipeline"]
    ratings = pd.read_csv(RATINGS_PATH)

    rows: list[dict] = []
    for _, result in results.iterrows():
        home = result["home_team"]
        away = result["away_team"]

        match_row = make_match_row(home, away, ratings, neutral=True)
        probabilities = predict_match(model, match_row)
        model_pick = max(probabilities, key=probabilities.get)

        rows.append(
            {
                **result.to_dict(),
                "model_pick": model_pick,
                "model_confidence": probabilities.get(model_pick, 0.0),
                "home_win_prob": probabilities.get("home_win", 0.0),
                "draw_prob": probabilities.get("draw", 0.0),
                "away_win_prob": probabilities.get("away_win", 0.0),
                "correct": model_pick == result["actual_outcome"],
            }
        )

    scored = pd.DataFrame(rows)
    scored.to_csv(OUT_PATH, index=False)
    return scored


def multiclass_brier_score(scored: pd.DataFrame) -> float:
    if scored.empty:
        return float("nan")
    probs = scored[["away_win_prob", "draw_prob", "home_win_prob"]].to_numpy(dtype=float)
    actual = scored["actual_outcome"].tolist()
    truth = np.zeros_like(probs)
    label_index = {label: idx for idx, label in enumerate(ORDERED_LABELS)}
    for row_idx, label in enumerate(actual):
        if label in label_index:
            truth[row_idx, label_index[label]] = 1.0
    return float(np.mean(np.sum((probs - truth) ** 2, axis=1)))


def summarize(scored: pd.DataFrame) -> None:
    if scored.empty:
        print("No completed matches available to score.")
        return

    y_true = scored["actual_outcome"].tolist()
    y_pred = scored["model_pick"].tolist()
    probability_matrix = scored[["away_win_prob", "draw_prob", "home_win_prob"]].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)
    try:
        loss = log_loss(y_true, probability_matrix, labels=ORDERED_LABELS)
    except Exception:
        loss = float("nan")
    brier = multiclass_brier_score(scored)

    print(f"Matches scored: {len(scored):,}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Log loss: {loss:.3f}")
    print(f"Brier score: {brier:.3f}")
    print()
    print(
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
                "correct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    summarize(score_predictions())
