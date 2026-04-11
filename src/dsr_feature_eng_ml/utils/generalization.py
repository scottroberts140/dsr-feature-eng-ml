"""Generalization assessment logic for model performance audits."""

from __future__ import annotations

from typing import Optional

from dsr_feature_eng_ml.enums import ModelGeneralization
from dsr_feature_eng_ml.prefs_instance import prefs


def calculate_generalization_status(
    score_train: Optional[float],
    score_valid: Optional[float],
    *,
    acceptable_gap: Optional[float] = None,
    large_gap: Optional[float] = None,
) -> ModelGeneralization:
    """
    Determine model generalization quality from train/valid scores.
    """
    if score_train is None or score_valid is None:
        return ModelGeneralization.PENDING

    # Late binding: Pull from prefs ONLY if no override was provided
    if acceptable_gap is None:
        acceptable_gap = prefs.acceptable_gap
    if large_gap is None:
        large_gap = prefs.large_gap

    # Use the rounding fix we implemented previously
    gap = round(score_train - score_valid, 4)

    if gap >= large_gap:
        return ModelGeneralization.OVERFIT
    if gap >= acceptable_gap:
        return ModelGeneralization.MARGINAL

    return ModelGeneralization.WELL_FIT
