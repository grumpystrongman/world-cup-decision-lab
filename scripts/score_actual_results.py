from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.models.predict import make_match_row, predict_match

RESULTS_PATH = REPO_ROOT / "data/actual/world_cup_results.csv"
MODEL_PATH = REPO_ROOT / "data/processed/match_model.joblib"
RATINGS_PATH = REPO_ROOT / "data/processed/team_ratings.csv"
OUT_PATH = REPO_ROOT / "data/actual/model_reality_check.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


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


def summarize(scored: pd.DataFrame) -> None:
    if scored.empty:
        print("No completed matches available to score.")
        return

    y_true = scored["actual_outcome"].tolist()
    y_pred = scored["model_pick"].tolist()
    labels = ["home_win", "draw", "away_win"]
    probability_matrix = scored[["home_win_prob", "draw_prob", "away_win_prob"]].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)
    try:
        loss = log_loss(y_true, probability_matrix, labels=labels)
    except Exception:
        loss = float("nan")

    print(f"Matches scored: {len(scored):,}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Log loss: {loss:.3f}")
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
