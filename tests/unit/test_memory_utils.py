from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dsr_feature_eng_ml.utils.memory import check_memory_risk, validate_n_jobs


def test_validate_n_jobs_logic():
    """Verify n_jobs is bounded correctly between 1 and CPU count."""
    with patch("os.cpu_count", return_value=8):
        # Test -1 (All cores)
        assert validate_n_jobs(-1) == 8

        # Test explicit count within bounds
        assert validate_n_jobs(4) == 4

        # Test explicit count exceeding bounds
        assert validate_n_jobs(12) == 8

        # Test safe fallback for 0 or negative (other than -1)
        assert validate_n_jobs(0) == 1
        assert validate_n_jobs(-5) == 1


def test_check_memory_risk_calculation():
    """Verify the heuristic math for peak memory estimation."""
    # 1. Create a mock DataFrame
    df = pd.DataFrame({"a": range(100000), "b": range(100000)})
    dataset_bytes = df.memory_usage(deep=True).sum()

    # 2. Mock ModelSpecification and its nested components
    mock_model = MagicMock()
    mock_model.model_type.tuning_multiplier = 2.0
    mock_model.model_dials.num_candidates = 10
    mock_model.total_fits = 50

    # 3. Mock psutil to control 'available memory'
    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 10**12  # 1 Terabyte

        risk, peak, available, multiplier = check_memory_risk(df, mock_model, n_jobs=2)

        # Use == instead of 'is' to handle NumPy boolean types correctly
        assert risk == False
        assert multiplier == 2.0
        assert peak > dataset_bytes


def test_check_memory_risk_high_risk_trigger():
    """Verify that the risk flag triggers when memory exceeds 85%."""
    df = pd.DataFrame({"a": range(1000)})
    mock_model = MagicMock()
    mock_model.model_type.tuning_multiplier = 10.0
    mock_model.model_dials.num_candidates = 100
    mock_model.total_fits = 500

    # Set available memory to be very low
    with patch("psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 100

        risk, *_ = check_memory_risk(df, mock_model, n_jobs=1)
        assert risk == True
