import itertools
import numpy as np
import pandas as pd

STAGES = ["qualifies_group", "round_of_32", "round_of_16", "quarterfinal", "semifinal", "final", "champion"]

def build_probability_cache(model, teams, ratings, adjustments=None):
    adjustments = adjustments or {}
    names = list(teams["team"])
    rating_lookup = dict(zip(ratings["team"], ratings["elo"]))

    pairs = []
    rows = []

    for home in names:
        for away in names:
            if home == away:
                continue

            home_elo = float(rating_lookup.get(home, 1500.0)) + float(adjustments.get(home, 0.0))
            away_elo = float(rating_lookup.get(away, 1500.0)) + float(adjustments.get(away, 0.0))
            expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))

            pairs.append((home, away))
            rows.append({
                "elo_diff_pre": home_elo - away_elo,
                "expected_home_pre": expected_home,
                "neutral": True,
                "recent_points_diff": 0.0,
                "recent_goal_diff_diff": 0.0,
            })

    prob_matrix = model.predict_proba(pd.DataFrame(rows))
    classes = list(model.classes_)

    home_idx = classes.index("home_win")
    away_idx = classes.index("away_win")

    cache = {}
    for pair, probs in zip(pairs, prob_matrix):
        home_p = float(probs[home_idx])
        away_p = float(probs[away_idx])
        total = home_p + away_p
        cache[pair] = 0.5 if total <= 0 else home_p / total

    return cache

def _home_wins(rng, cache, home, away):
    return rng.random() < cache[(home, away)]

def _qualifiers_once(groups, fixtures, ratings_lookup, rng, cache):
    qualifiers = []
    thirds = []

    for group, members in groups.items():
        points = {team: 0 for team in members}
        wins = {team: 0 for team in members}

        for home, away in fixtures[group]:
            if _home_wins(rng, cache, home, away):
                points[home] += 3
                wins[home] += 1
            else:
                points[away] += 3
                wins[away] += 1

        ranked = sorted(
            members,
            key=lambda t: (points[t], wins[t], ratings_lookup.get(t, 1500.0)),
            reverse=True
        )

        qualifiers.extend(ranked[:2])

        if len(ranked) >= 3:
            third = ranked[2]
            thirds.append((third, points[third], wins[third], ratings_lookup.get(third, 1500.0)))

    best_thirds = [
        row[0]
        for row in sorted(thirds, key=lambda x: (x[1], x[2], x[3]), reverse=True)[:8]
    ]

    qualifiers.extend(best_thirds)
    return qualifiers[:32]

def _knockout_once(qualifiers, rng, cache):
    current = list(qualifiers)
    rng.shuffle(current)

    progress = {}
    round_names = {
        32: "round_of_32",
        16: "round_of_16",
        8: "quarterfinal",
        4: "semifinal",
        2: "final",
    }

    while len(current) > 1:
        stage = round_names.get(len(current), f"round_{len(current)}")

        for team in current:
            progress[team] = stage

        winners = []

        for idx in range(0, len(current), 2):
            home = current[idx]
            away = current[idx + 1]
            winners.append(home if _home_wins(rng, cache, home, away) else away)

        current = winners

    progress[current[0]] = "champion"
    return progress

def run_tournament_simulation(model, teams, ratings, n_sims=10000, seed=42, adjustments=None, progress_callback=None):
    rng = np.random.default_rng(seed)

    teams = teams.copy().reset_index(drop=True)
    ratings_lookup = dict(zip(ratings["team"], ratings["elo"]))

    groups = {
        str(group): list(group_df["team"])
        for group, group_df in teams.groupby("group")
    }

    fixtures = {
        group: list(itertools.combinations(members, 2))
        for group, members in groups.items()
    }

    cache = build_probability_cache(model, teams, ratings, adjustments=adjustments)

    counters = {
        team: {stage: 0 for stage in STAGES}
        for team in teams["team"]
    }

    progress_every = max(1, n_sims // 100)

    for sim_index in range(n_sims):
        qualifiers = _qualifiers_once(groups, fixtures, ratings_lookup, rng, cache)

        for team in qualifiers:
            counters[team]["qualifies_group"] += 1

        progress = _knockout_once(qualifiers, rng, cache)

        for team, stage in progress.items():
            counters[team][stage] += 1

        if progress_callback and (sim_index + 1) % progress_every == 0:
            progress_callback((sim_index + 1) / n_sims)

    result = pd.DataFrame([
        {"team": team, **counts}
        for team, counts in counters.items()
    ])

    for stage in STAGES:
        result[stage] = result[stage] / n_sims

    return result.sort_values("champion", ascending=False).reset_index(drop=True)
