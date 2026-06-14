import pandas as pd

def feature_importance_from_model(pipeline):
    model = pipeline.named_steps["model"]
    features = list(pipeline.named_steps["prep"].feature_names_in_)
    values = getattr(model, "feature_importances_", [])
    return (
        pd.DataFrame({"feature": features, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

def plain_english_match_explanation(home_team, away_team, probabilities, match_row):
    home_prob = probabilities.get("home_win", 0.0)
    away_prob = probabilities.get("away_win", 0.0)
    draw_prob = probabilities.get("draw", 0.0)
    favorite = home_team if home_prob >= away_prob else away_team
    confidence = max(home_prob, away_prob)
    elo_diff = float(match_row.iloc[0]["elo_diff_pre"])

    if abs(elo_diff) < 40:
        strength = "The Elo gap is narrow, so this is a volatile matchup."
    elif elo_diff > 0:
        strength = f"{home_team} has the stronger pre-match Elo profile."
    else:
        strength = f"{away_team} has the stronger pre-match Elo profile."

    return (
        f"{favorite} is favored at {confidence:.1%}. "
        f"The draw probability is {draw_prob:.1%}. "
        f"{strength} Scenario controls show how sensitive the model is to team-strength assumptions."
    )

def executive_tournament_summary(simulation_df, scenario_label="baseline"):
    top = simulation_df.head(5).copy()
    leader = top.iloc[0]
    challengers = ", ".join(top.iloc[1:5]["team"].tolist())
    return (
        f"Under the {scenario_label} scenario, {leader['team']} has the strongest simulated title path "
        f"at {leader['champion']:.1%}. The next tier of challengers is {challengers}. "
        "This is a decision-intelligence view, not a deterministic forecast."
    )
