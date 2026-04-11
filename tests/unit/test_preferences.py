import json
from pathlib import Path

import pytest

from dsr_feature_eng_ml.prefs_instance import prefs


@pytest.fixture(autouse=True)
def reset_prefs():
    """Ensure preferences are reset to library defaults after every test."""
    yield
    prefs.reset_defaults()


def test_singleton_integrity():
    """Verify that Preferences follows the singleton pattern."""
    from dsr_feature_eng_ml.preferences import Preferences

    new_prefs = Preferences()
    assert new_prefs is prefs
    assert id(new_prefs) == id(prefs)


def test_cv_verbose_clamping():
    """Verify that cv_verbose is clamped between 0 and 3."""
    prefs.cv_verbose = 5
    assert prefs.cv_verbose == 3

    prefs.cv_verbose = -1
    assert prefs.cv_verbose == 0

    prefs.cv_verbose = 2
    assert prefs.cv_verbose == 2


def test_update_valid_attributes():
    """Test updating multiple attributes at once."""
    prefs.update(acceptable_gap=0.10, random_state=100)
    assert prefs.acceptable_gap == 0.10
    assert prefs.random_state == 100


def test_update_invalid_attribute():
    """Verify that updating a non-existent attribute raises AttributeError."""
    with pytest.raises(AttributeError):
        prefs.update(non_existent_setting=True)


def test_reset_defaults():
    """Verify that reset_defaults restores the original library state."""
    original_gap = prefs.acceptable_gap
    prefs.acceptable_gap = 0.50

    prefs.reset_defaults()
    assert prefs.acceptable_gap == original_gap


def test_json_persistence(tmp_path):
    """Verify saving and loading from a JSON file."""
    test_file = tmp_path / "prefs.json"
    original_random_state = prefs.random_state
    new_state = 999

    prefs.random_state = new_state
    prefs.save_to_json(test_file)

    # Reset to original
    prefs.random_state = original_random_state

    # Load from file
    prefs.load_from_json(test_file)
    assert prefs.random_state == new_state


def test_repr_output():
    """Verify the string representation contains key fields."""
    output = repr(prefs)
    assert "acceptable_gap" in output
    assert "random_state" in output
