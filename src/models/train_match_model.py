from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.draw_aware_model import DrawAwareTwoStageModel

FEATURES = [
    "elo_diff_pre",
    "expected_home_pre",
    "neutral",
    "recent_points_diff",
    "recent_goal_diff_diff",
    "form_volatility_diff",
    "abs_elo_diff_pre",
    "expected_uncertainty",
    "close_match_flag",
    "draw_likelihood",
    "upset_risk",
]


def _make_preprocessor():
    return ColumnTransformer([("num", StandardScaler(), FEATURES)], remainder="drop")


def _make_calibrated_forest(max_depth=7, min_samples_leaf=10, n_estimators=600):
    base_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    return CalibratedClassifierCV(base_classifier, method="sigmoid", cv=3)


def _make_pipeline(max_depth=7, min_samples_leaf=10, n_estimators=600):
    return Pipeline(
        [
            ("prep", _make_preprocessor()),
            ("model", _make_calibrated_forest(max_depth=max_depth, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)),
        ]
    )


def train_model(features_df, output_path="data/processed/match_model.joblib"):
    df = features_df.dropna(subset=["target"]).copy()
    if len(df) < 50:
        raise ValueError("Need at least 50 matches to train a draw-aware model. Use --use-sample or add real results.csv.")

    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0

    X = df[FEATURES]
    y = df["target"]
    split_index = max(int(len(df) * 0.8), 1)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    multiclass_pipeline = _make_pipeline(max_depth=7, min_samples_leaf=10, n_estimators=600)
    multiclass_pipeline.fit(X_train, y_train)

    draw_y_train = y_train.apply(lambda value: "draw" if value == "draw" else "decisive")
    draw_pipeline = _make_pipeline(max_depth=6, min_samples_leaf=12, n_estimators=500)
    draw_pipeline.fit(X_train, draw_y_train)

    decisive_mask = y_train != "draw"
    decisive_pipeline = _make_pipeline(max_depth=6, min_samples_leaf=10, n_estimators=500)
    decisive_pipeline.fit(X_train[decisive_mask], y_train[decisive_mask])

    pipeline = DrawAwareTwoStageModel(
        multiclass_pipeline=multiclass_pipeline,
        draw_pipeline=draw_pipeline,
        decisive_pipeline=decisive_pipeline,
        two_stage_weight=0.72,
        min_draw_probability=0.12,
        close_match_draw_floor=0.24,
    )

    probabilities = pipeline.predict_proba(X_test)
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=list(pipeline.classes_))),
        "classes": list(pipeline.classes_),
        "features": FEATURES,
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "probability_calibration": "DrawAwareTwoStageModel: draw detector + decisive winner model + calibrated multiclass stabilizer",
        "model_note": "Explicitly models draw risk before allocating decisive win probability.",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, output_path)
    return metrics


def load_model(path="data/processed/match_model.joblib"):
    return joblib.load(path)["pipeline"]
