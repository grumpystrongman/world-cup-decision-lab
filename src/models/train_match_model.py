from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def train_model(features_df, output_path="data/processed/match_model.joblib"):
    df = features_df.dropna(subset=["target"]).copy()
    if len(df) < 20:
        raise ValueError("Need at least 20 matches to train. Use --use-sample for demo or add real results.csv.")

    for feature in FEATURES:
        if feature not in df.columns:
            df[feature] = 0.0

    X = df[FEATURES]
    y = df["target"]
    split_index = max(int(len(df) * 0.8), 1)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    preprocessor = ColumnTransformer([("num", StandardScaler(), FEATURES)], remainder="drop")
    base_classifier = RandomForestClassifier(
        n_estimators=600,
        max_depth=7,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    classifier = CalibratedClassifierCV(base_classifier, method="sigmoid", cv=3)
    pipeline = Pipeline([("prep", preprocessor), ("model", classifier)])
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)
    predictions = pipeline.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=pipeline.classes_)),
        "classes": list(pipeline.classes_),
        "features": FEATURES,
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "probability_calibration": "CalibratedClassifierCV sigmoid cv=3",
        "model_note": "Includes draw/uncertainty/upset-risk features to reduce overconfident misses.",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics}, output_path)
    return metrics


def load_model(path="data/processed/match_model.joblib"):
    return joblib.load(path)["pipeline"]
