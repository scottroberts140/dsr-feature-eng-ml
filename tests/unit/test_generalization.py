from typing import Optional

import pytest

from dsr_feature_eng_ml.enums import ModelGeneralization
from dsr_feature_eng_ml.prefs_instance import prefs
from dsr_feature_eng_ml.utils.generalization import calculate_generalization_status


@pytest.fixture(autouse=True)
def reset_prefs():
    """Ensure preference thresholds are at defaults for each test."""
    prefs.reset_defaults()
    yield


def test_pending_status():
    """Verify that status is PENDING if scores are missing."""
    assert calculate_generalization_status(None, 0.8) == ModelGeneralization.PENDING
    assert calculate_generalization_status(0.8, None) == ModelGeneralization.PENDING
    assert calculate_generalization_status(None, None) == ModelGeneralization.PENDING


def test_well_fit_status():
    """Verify WELL_FIT for gaps below the acceptable threshold."""
    # Gap of 0.01 is less than default acceptable_gap (0.02)
    assert calculate_generalization_status(0.81, 0.80) == ModelGeneralization.WELL_FIT

    # Negative gap (Validation > Train) should still be WELL_FIT
    assert calculate_generalization_status(0.75, 0.80) == ModelGeneralization.WELL_FIT


def test_marginal_status():
    """Verify MARGINAL for gaps between acceptable and large thresholds."""
    # Use exact values that are less prone to binary floating point noise
    # Or rely on the 'round()' fix in the source code.
    assert calculate_generalization_status(0.83, 0.80) == ModelGeneralization.MARGINAL

    # This will now pass with the rounding fix in the source
    assert calculate_generalization_status(0.82, 0.80) == ModelGeneralization.MARGINAL


def test_overfit_status():
    """Verify OVERFIT for gaps exceeding the large threshold."""
    # Gap 0.06 > 0.05
    assert calculate_generalization_status(0.86, 0.80) == ModelGeneralization.OVERFIT

    # Boundary check: Exactly the large_gap should be OVERFIT
    assert calculate_generalization_status(0.85, 0.80) == ModelGeneralization.OVERFIT


def test_explicit_threshold_overrides():
    """Verify that passing explicit thresholds overrides prefs."""
    # Normally 0.03 is MARGINAL, but with these overrides, it's WELL_FIT
    status = calculate_generalization_status(
        0.83, 0.80, acceptable_gap=0.05, large_gap=0.10
    )
    assert status == ModelGeneralization.WELL_FIT


def test_prefs_integration():
    """Verify logic reacts correctly when global prefs are changed."""
    prefs.acceptable_gap = 0.005  # Tighten threshold
    # Gap of 0.01 is now Marginal instead of Well-Fit
    assert calculate_generalization_status(0.81, 0.80) == ModelGeneralization.MARGINAL
