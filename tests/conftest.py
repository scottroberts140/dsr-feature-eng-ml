from pathlib import Path

import pytest
from dsr_files.parquet_handler import load_parquet

from dsr_feature_eng_ml.enums import BalancingStrategy, ScoringMetric, TaskType


@pytest.fixture
def mini_taxi_df():
    """Load the 1000-row test sample for audit verification."""
    data_path = Path(__file__).parent / "data" / "YellowTaxi_202510.parquet"
    df, _ = load_parquet(data_path)
    return df


@pytest.fixture
def regression_config():
    """Standard audit configuration for regression tasks."""
    return {
        "cv": 5,
        "balancing_strategy": BalancingStrategy.NONE,
        "task_type": TaskType.REGRESSION,
        "scoring": ScoringMetric.R2,
    }
