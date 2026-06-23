from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class DrawAwareTwoStageModel(BaseEstimator, ClassifierMixin):
    """Blend a draw detector with a decisive-outcome winner model.

    International soccer has a persistent draw problem. A single multiclass model
    often underestimates draws because favorites dominate the win/loss signal.
    This wrapper explicitly models:
      1. draw vs decisive
      2. home win vs away win when the match is decisive
      3. a stabilizing multiclass probability model
    """

    classes_ = np.array(["away_win", "draw", "home_win"], dtype=object)

    def __init__(
        self,
        multiclass_pipeline,
        draw_pipeline,
        decisive_pipeline,
        two_stage_weight: float = 0.72,
        min_draw_probability: float = 0.12,
        close_match_draw_floor: float = 0.24,
    ):
        self.multiclass_pipeline = multiclass_pipeline
        self.draw_pipeline = draw_pipeline
        self.decisive_pipeline = decisive_pipeline
        self.two_stage_weight = two_stage_weight
        self.min_draw_probability = min_draw_probability
        self.close_match_draw_floor = close_match_draw_floor

    @staticmethod
    def _positive_class_probability(pipeline, X, positive_label):
        probs = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        if positive_label not in classes:
            return np.zeros(len(X), dtype=float)
        return probs[:, classes.index(positive_label)].astype(float)

    @staticmethod
    def _class_probability(pipeline, X, label, default=0.0):
        probs = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        if label not in classes:
            return np.full(len(X), default, dtype=float)
        return probs[:, classes.index(label)].astype(float)

    def predict_proba(self, X):
        X_df = pd.DataFrame(X).copy()

        multi_away = self._class_probability(self.multiclass_pipeline, X_df, "away_win", 0.30)
        multi_draw = self._class_probability(self.multiclass_pipeline, X_df, "draw", 0.27)
        multi_home = self._class_probability(self.multiclass_pipeline, X_df, "home_win", 0.43)
        multiclass_probs = np.column_stack([multi_away, multi_draw, multi_home])

        draw_prob = self._positive_class_probability(self.draw_pipeline, X_df, "draw")
        home_cond = self._positive_class_probability(self.decisive_pipeline, X_df, "home_win")
        away_cond = 1.0 - home_cond

        two_stage_probs = np.column_stack([
            (1.0 - draw_prob) * away_cond,
            draw_prob,
            (1.0 - draw_prob) * home_cond,
        ])

        combined = (self.two_stage_weight * two_stage_probs) + ((1.0 - self.two_stage_weight) * multiclass_probs)

        if "draw_likelihood" in X_df.columns:
            draw_likelihood = X_df["draw_likelihood"].astype(float).clip(0, 1).to_numpy()
            dynamic_floor = self.min_draw_probability + (self.close_match_draw_floor - self.min_draw_probability) * draw_likelihood
            for idx, floor_value in enumerate(dynamic_floor):
                if combined[idx, 1] < floor_value:
                    deficit = floor_value - combined[idx, 1]
                    non_draw_total = combined[idx, 0] + combined[idx, 2]
                    if non_draw_total > 0:
                        combined[idx, 0] -= deficit * (combined[idx, 0] / non_draw_total)
                        combined[idx, 2] -= deficit * (combined[idx, 2] / non_draw_total)
                    combined[idx, 1] = floor_value

        combined = np.clip(combined, 1e-6, 1.0)
        combined = combined / combined.sum(axis=1, keepdims=True)
        return combined

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
