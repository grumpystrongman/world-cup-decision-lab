import itertools
import numpy as np
import pandas as pd
from src.models.predict import make_match_row, predict_match, sample_outcome


def build_group_fixtures(teams):
    fixtures = []
    for group, group_df in teams.groupby("group"):
        for home, away in itertools.combinations(list(group_df["team"]), 2):
            fixtures.append((str(group), home, away))
    return fixtures


def points_for_outcome(outcome):
    if outcome == "home_win":
        return 3, 0
    if outcome == "away_win":
        return 0, 3
    return 1, 1


def simulate_group_stage(model, teams, ratings, rng, adjustments=None):
    standings = {
        row.team: {"team": row.team, "group": row.group, "points": 0, "wins": 0}
        for row in teams.itertuples(index=False)
    }
    for _, home, away in build_group_fixtures(teams):
        row = make_match_row(home, away, ratings, neutral=True, team_adjustments=adjustments)
        outcome = sample_outcome(predict_match(model, row), rng)
        home_points, away_points = points_for_outcome(outcome)
        standings[home]["points"] += home_points
        standings[away]["points"] += away_points
        if outcome == "home_win":
            standings[home]["wins"] += 1
        elif outcome == "away_win":
            standings[away]["wins"] += 1
    table = pd.DataFrame(standings.values())
    rating_map = dict(zip(ratings["team"], ratings["elo"]))
    table["elo"] = table["team"].map(rating_map).fillna(1500.0)
    return table.sort_values(["group", "points", "wins", "elo"], ascending=[True, False, False, False])


def choose_group_qualifiers(group_table, teams_per_group=2):
    qualifiers = []
    for _, group_df in group_table.groupby("group"):
        qualifiers.extend(list(group_df.head(teams_per_group)["team"]))
    return qualifiers


def simulate_knockout(model, qualifiers, ratings, rng, adjustments=None):
    current = qualifiers[:]
    rng.shuffle(current)
    progress = {}
    round_names = {16: "round_of_16", 8: "quarterfinal", 4: "semifinal", 2: "final"}
    while len(current) > 1:
        round_name = round_names.get(len(current), f"round_{len(current)}")
        winners = []
        for idx in range(0, len(current), 2):
            home, away = current[idx], current[idx + 1]
            row = make_match_row(home, away, ratings, neutral=True, team_adjustments=adjustments)
            probs = predict_match(model, row)
            no_draw = {"home_win": probs.get("home_win", 0.0), "away_win": probs.get("away_win", 0.0)}
            total = sum(no_draw.values()) or 1.0
            no_draw = {key: value / total for key, value in no_draw.items()}
            outcome = sample_outcome(no_draw, rng)
            winners.append(home if outcome == "home_win" else away)
        for winner in winners:
            progress[winner] = round_name
        current = winners
    progress[current[0]] = "champion"
    return progress


def run_tournament_simulation(model, teams, ratings, n_sims=10000, seed=42, adjustments=None):
    rng = np.random.default_rng(seed)
    stages = ["qualifies_group", "round_of_16", "quarterfinal", "semifinal", "final", "champion"]
    counters = {team: {"team": team, **{stage: 0 for stage in stages}} for team in teams["team"]}
    for _ in range(n_sims):
        group_table = simulate_group_stage(model, teams, ratings, rng, adjustments=adjustments)
        qualifiers = choose_group_qualifiers(group_table)
        for team in qualifiers:
            counters[team]["qualifies_group"] += 1
        if len(qualifiers) < 2:
            continue
        power_of_two = 2 ** int(np.floor(np.log2(len(qualifiers))))
        qualifiers = qualifiers[:power_of_two]
        progress = simulate_knockout(model, qualifiers, ratings, rng, adjustments=adjustments)
        for team, reached_stage in progress.items():
            counters[team][reached_stage] += 1
    result = pd.DataFrame(counters.values())
    for stage in stages:
        result[stage] = result[stage] / n_sims
    return result.sort_values("champion", ascending=False).reset_index(drop=True)
