import numpy as np
import pandas as pd
from src.features.elo import expected_score


def make_match_row(
    home_team,
    away_team,
    ratings,
    neutral=True,
    home_advantage=60.0,
    home_recent_points=1.5,
    away_recent_points=1.5,
    home_recent_goal_diff=0.0,
    away_recent_goal_diff=0.0,
    home_form_volatility=0.0,
    away_form_volatility=0.0,
    team_adjustments=None,
):
    team_adjustments = team_adjustments or {}
    rating_map = dict(zip(ratings["team"], ratings["elo"]))
    home_elo = float(rating_map.get(home_team, 1500.0)) + float(team_adjustments.get(home_team, 0.0))
    away_elo = float(rating_map.get(away_team, 1500.0)) + float(team_adjustments.get(away_team, 0.0))
    context_home_elo = home_elo + (0.0 if neutral else home_advantage)
    elo_diff = context_home_elo - away_elo
    expected_home = expected_score(context_home_elo, away_elo)
    expected_uncertainty = 1.0 - abs(expected_home - 0.5) * 2.0

    return pd.DataFrame([
        {
            "elo_diff_pre": elo_diff,
            "expected_home_pre": expected_home,
            "neutral": bool(neutral),
            "recent_points_diff": home_recent_points - away_recent_points,
            "recent_goal_diff_diff": home_recent_goal_diff - away_recent_goal_diff,
            "form_volatility_diff": home_form_volatility - away_form_volatility,
            "abs_elo_diff_pre": abs(elo_diff),
            "expected_uncertainty": expected_uncertainty,
            "close_match_flag": 1 if abs(elo_diff) < 75 else 0,
            "draw_likelihood": max(0.0, min(1.0, expected_uncertainty)),
            "upset_risk": max(0.0, min(1.0, 1.0 - abs(elo_diff) / 400.0)),
        }
    ])


def predict_match(model, match_row):
    probabilities = model.predict_proba(match_row)[0]
    return {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}


def sample_outcome(probabilities, rng):
    labels = list(probabilities.keys())
    probs = np.array([probabilities[label] for label in labels], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(labels, p=probs))
