"""Generalization assessment logic for model performance audits."""

from __future__ import annotations

from dsr_feature_eng_ml.enums import ModelGeneralization
from dsr_feature_eng_ml.prefs_instance import prefs


def calculate_generalization_status(
    score_train: float | None,
    score_valid: float | None,
    *,
    acceptable_gap: float | None = None,
    large_gap: float | None = None,
) -> ModelGeneralization:
    """
    Determine model generalization quality from train/validation scores.

    Parameters
    ----------
    score_train : float or None
        The model's score on the training set. If None, returns PENDING.
    score_valid : float or None
        The model's score on the validation set. If None, returns PENDING.
    acceptable_gap : float or None, optional
        Maximum gap considered Well-Fit. Defaults to ``prefs.acceptable_gap``.
    large_gap : float or None, optional
        Gap above which the model is classified as Overfit. Defaults to
        ``prefs.large_gap``.

    Returns
    -------
    ModelGeneralization
        PENDING if either score is None; otherwise WELL_FIT, MARGINAL, or
        OVERFIT based on ``round(score_train - score_valid, 4)``.
    """
    if score_train is None or score_valid is None:
        return ModelGeneralization.PENDING

    # Late binding: pull from prefs only if no override was provided
    if acceptable_gap is None:
        acceptable_gap = prefs.acceptable_gap
    if large_gap is None:
        large_gap = prefs.large_gap

    gap = round(score_train - score_valid, 4)

    if gap >= large_gap:
        return ModelGeneralization.OVERFIT
    if gap >= acceptable_gap:
        return ModelGeneralization.MARGINAL

    return ModelGeneralization.WELL_FIT
