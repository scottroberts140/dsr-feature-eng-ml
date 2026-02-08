from __future__ import annotations

from typing import Optional

from dsr_feature_eng_ml.preferences import prefs
from dsr_feature_eng_ml.enums import ModelGeneralization


def calculate_generalization_status(
    score_train: Optional[float],
    score_valid: Optional[float],
    *,
    acceptable_gap: float = prefs.acceptable_gap,
    large_gap: float = prefs.large_gap,
) -> ModelGeneralization:
    """Determine model generalization quality from train/valid scores.

    Returns:
        ModelGeneralization: Good/Acceptable/Overfitted/Undefined based on gaps.
    """
    if score_train is None or score_valid is None:
        return ModelGeneralization.PENDING

    gap = score_train - score_valid
    if gap >= large_gap:
        return ModelGeneralization.OVERFIT
    if gap >= acceptable_gap:
        return ModelGeneralization.MARGINAL
    return ModelGeneralization.WELL_FIT
