import numpy as np
import pandas as pd
from src.features.elo import expected_score

OUTCOME_PRIOR = {
    "home_win": 0.43,
    "draw": 0.27,
    "away_win": 0.30,
}


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


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
    recent_points_diff = home_recent_points - away_recent_points
    recent_goal_diff_diff = home_recent_goal_diff - away_recent_goal_diff
    form_volatility_diff = home_form_volatility - away_form_volatility
    form_signal = recent_points_diff + 0.35 * recent_goal_diff_diff
    elo_signal = elo_diff / 400.0
    abs_elo = abs(elo_diff)
    upset_risk = _clamp(1.0 - abs_elo / 400.0)

    return pd.DataFrame([
        {
            "elo_diff_pre": elo_diff,
            "expected_home_pre": expected_home,
            "neutral": bool(neutral),
            "recent_points_diff": recent_points_diff,
            "recent_goal_diff_diff": recent_goal_diff_diff,
            "form_volatility_diff": form_volatility_diff,
            "goal_trend_diff": 0.0,
            "abs_elo_diff_pre": abs_elo,
            "expected_uncertainty": _clamp(expected_uncertainty),
            "close_match_flag": 1 if abs_elo < 75 else 0,
            "draw_likelihood": _clamp(expected_uncertainty),
            "upset_risk": upset_risk,
            "form_spike": _clamp(max(0.0, abs(form_signal) - 1.0) / 2.5),
            "form_crash": _clamp(max(0.0, -form_signal - 1.0) / 2.5),
            "rating_form_disagreement": _clamp(abs(elo_signal - form_signal / 3.0)),
            "favorite_fragility": _clamp((1.0 if abs_elo > 120 else 0.4) * upset_risk * (1.0 + abs(form_volatility_diff) / 2.0)),
            "low_scoring_draw_proxy": _clamp(expected_uncertainty * (1.0 - min(1.0, abs(recent_goal_diff_diff) / 3.0))),
            "tactical_mismatch_proxy": _clamp(abs(recent_goal_diff_diff - elo_signal) / 3.0),
            "pressure_discontinuity_proxy": _clamp(expected_uncertainty * (1.0 + upset_risk) / 2.0),
        }
    ])


def _smooth_probabilities(probability_map, match_row=None, shrinkage=0.12):
    labels = list(probability_map.keys())
    smoothed = {}
    for label in labels:
        prior = OUTCOME_PRIOR.get(label, 1.0 / max(1, len(labels)))
        smoothed[label] = (1.0 - shrinkage) * float(probability_map[label]) + shrinkage * prior

    if match_row is not None and "draw" in smoothed:
        try:
            uncertainty = float(match_row.iloc[0].get("draw_likelihood", 0.0))
            pressure = float(match_row.iloc[0].get("pressure_discontinuity_proxy", 0.0))
            low_scoring = float(match_row.iloc[0].get("low_scoring_draw_proxy", 0.0))
        except Exception:
            uncertainty = pressure = low_scoring = 0.0
        draw_lift = 0.035 * _clamp(uncertainty) + 0.025 * _clamp(pressure) + 0.020 * _clamp(low_scoring)
        non_draw_labels = [label for label in labels if label != "draw"]
        non_draw_total = sum(smoothed[label] for label in non_draw_labels) or 1.0
        for label in non_draw_labels:
            smoothed[label] -= draw_lift * (smoothed[label] / non_draw_total)
        smoothed["draw"] += draw_lift

    total = sum(max(0.0, value) for value in smoothed.values()) or 1.0
    return {label: max(0.0, value) / total for label, value in smoothed.items()}


def predict_match(model, match_row, smooth=True):
    probabilities = model.predict_proba(match_row)[0]
    raw = {str(label): float(prob) for label, prob in zip(model.classes_, probabilities)}
    if not smooth:
        return raw
    return _smooth_probabilities(raw, match_row=match_row)


def sample_outcome(probabilities, rng):
    labels = list(probabilities.keys())
    probs = np.array([probabilities[label] for label in labels], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(labels, p=probs))
