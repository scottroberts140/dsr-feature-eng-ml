import dataclasses

import pandas as pd
import pytest
from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.evaluation import ModelAuditSummary
from dsr_feature_eng_ml.evaluation.model_auditor import (
    AuditLogger,
    ModelAuditor,
    ModelAuditorConfig,
)
from dsr_feature_eng_ml.evaluation.schema import (
    DataSplits,
    FeatureMetadata,
    ModelAuditorConfig,
    ModelConfiguration,
)
from dsr_feature_eng_ml.models.lasso_regression import LassoParams, LassoRegression
from dsr_feature_eng_ml.models.random_forest import (
    RandomForestRegressorModel,
    RandomForestRegressorParams,
)


def test_audit_logger_redirection(tmp_path):
    """Verify that the logger captures terminal output into a file."""
    log_file = tmp_path / "audit_test.log"
    test_message = "Audit started for Taxi Dataset...\n"

    # 1. Use the context manager to ensure proper cleanup
    with AuditLogger(log_file) as logger:
        logger.write(test_message)
        logger.flush()  #

    # 2. Verify the file content
    assert log_file.exists()
    assert log_file.read_text(encoding="utf-8") == test_message


def test_audit_logger_context_management(tmp_path):
    """Ensure the log file is closed automatically upon exit."""
    log_file = tmp_path / "context_test.log"

    with AuditLogger(log_file) as logger:
        assert not logger.log.closed  #

    assert logger.log.closed  #


def test_audit_logger_manual_flush(tmp_path):
    """Verify that flush correctly clears buffers for both terminal and file."""
    log_file = tmp_path / "flush_test.log"
    logger = AuditLogger(log_file)

    try:
        logger.write("Telemetry captured.")
        # Flush should be called during high-risk operations (like Fit)
        logger.flush()
        assert log_file.read_text(encoding="utf-8") == "Telemetry captured."
    finally:
        logger.close()  #


def test_auditor_phase_increment():
    """Verify that each new auditor instance increments the global phase number."""
    initial_phase = ModelAuditor.phase_number

    # Instantiate two auditors
    config = ModelAuditorConfig(data_splits=DataSplits.empty(), dataset_name="Test")
    auditor_1 = ModelAuditor(config)
    auditor_2 = ModelAuditor(config)

    assert auditor_1.current_phase == initial_phase + 1
    assert auditor_2.current_phase == initial_phase + 2


def test_auditor_trace_logging(mini_taxi_df, tmp_path):
    """Verify that run_audit creates a log file and mirrors terminal output."""
    # 1. Manually build FeatureMetadata for the columns you want to audit
    # This ensures the auditor identifies these as 'used_in_fit'
    features = FeatureMetadata.from_df(
        df=mini_taxi_df,
        exclude_from_fit={
            "fare_amount",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "store_and_fwd_flag",
        },
    )

    # 2. Pass the features dictionary to the config
    config = ModelAuditorConfig.from_dataset(
        dataset=mini_taxi_df,
        original_row_count=1000,
        target_column="fare_amount",
        dataset_name="Taxi_Log_Test",
        cv=2,
        model_classes=[RandomForestRegressorModel],
        task_type=TaskType.REGRESSION,
        features=features,
    )

    auditor = ModelAuditor(config)

    # 3. Run audit
    auditor.run_audit(optimize=False, save_path=tmp_path)

    # Check for log persistence
    log_files = list(tmp_path.glob("audit_trace_*.log"))
    assert len(log_files) == 1
    assert "Auditing RANDOM_FOREST_REGRESSOR" in log_files[0].read_text()


def test_model_viability_pruning(mini_taxi_df):
    """
    Verify that models with a large score gap are pruned from the config.
    """
    features = FeatureMetadata.from_df(
        df=mini_taxi_df,
        exclude_from_fit={
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "store_and_fwd_flag",
        },
    )

    config = ModelAuditorConfig.from_dataset(
        dataset=mini_taxi_df,
        original_row_count=len(mini_taxi_df),
        target_column="fare_amount",
        dataset_name="Pruning_Test",
        cv=5,
        model_classes=[RandomForestRegressorModel, LassoRegression],
        task_type=TaskType.REGRESSION,
        features=features,
    )
    config.viable_score_gap = 0.01

    auditor = ModelAuditor(config)
    auditor.run_audit(optimize=False)

    # 1. Verify pruning occurred
    # One model should be gone if the gap between them > 0.01
    print(f"Remaining models: {[m.model_type.name for m in config.models_to_run]}")
    assert len(config.models_to_run) == 1

    # 2. Assert that the remaining model is either Lasso or RandomForest
    # In your current run, Lasso is the winner
    winner = config.models_to_run[0]
    assert isinstance(winner, (RandomForestRegressorModel, LassoRegression))


@pytest.fixture
def populated_summary(mini_taxi_df):
    """Create a summary with two competing models (RFR vs Lasso)."""
    # 1. Create a Winning Model (RFR)
    m1 = ModelConfiguration(
        id="01",
        model_type=ModelType.RANDOM_FOREST_REGRESSOR,
        task_type=TaskType.REGRESSION,
        score_val=0.8145,
        preds_val=pd.Series([20.0, 15.0]),  # Mock predictions
        balancing_strategy=BalancingStrategy.NONE,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        cv=5,
        scoring=ScoringMetric.R2,
        n_jobs=1,
        n_iter=10,
        model_params=RandomForestRegressorParams(
            scoring=ScoringMetric.R2, random_state=75
        ),
    )
    # 2. Create a Losing Model (Lasso)
    m2 = ModelConfiguration(
        id="02",
        model_type=ModelType.LASSO,
        task_type=TaskType.REGRESSION,
        score_val=0.7500,
        balancing_strategy=BalancingStrategy.NONE,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        cv=5,
        scoring=ScoringMetric.R2,
        n_jobs=1,
        n_iter=10,
        model_params=LassoParams(),
    )
    target = "fare_amount"
    features = [c for c in mini_taxi_df.columns if c != target]

    return ModelAuditSummary(
        data_splits=DataSplits.from_data_source(
            mini_taxi_df,
            features_to_include=features,
            target_column=target,
            test_size=0.2,
            valid_size=0.2,
            original_row_count=len(mini_taxi_df),
            random_state=75,
        ),
        results=[m1, m2],
        dataset_name="Taxi Audit Test",
        original_row_count=1000,
    )


def test_best_model_selection(populated_summary):
    """Verify that best_overall_model correctly identifies the RFR winner."""
    # Uses ModelConfiguration.__lt__ (based on score_val)
    best = populated_summary.best_overall_model
    assert best is not None
    assert best.id == "01"
    assert best.score_val == 0.8145


def test_anomaly_context_capture(populated_summary):
    """
    Verify high-error row identification.
    Matches the 'Anomaly Risk Profile'.
    """
    # Mock data splits target vs predictions
    # If target is [20, 15] and preds are [20, 50], index 1 is an anomaly
    # The number of values in the target column has to be the same as the
    # number of rows in the dataframe.
    df_len = len(populated_summary.data_splits.val_target)
    target_values = pd.Series(20.0).repeat(df_len - 1)
    target_values[1] = 15.0
    preds_values = pd.Series(20.0).repeat(df_len - 1)
    preds_values[1] = 50.0
    populated_summary.data_splits = dataclasses.replace(
        populated_summary.data_splits, val_target=target_values
    )
    populated_summary.results[0] = dataclasses.replace(
        populated_summary.results[0],
        preds_val=pd.Series(
            preds_values, index=populated_summary.data_splits.val_target.index
        ),
    )

    populated_summary.capture_anomaly_context()

    # Anomaly data should exist and be sorted by absolute error
    assert populated_summary.anomaly_data is not None
    assert len(populated_summary.anomaly_data) > 0
    # The error (abs(15 - 50) = 35) should be the top record
    assert populated_summary.anomaly_data.iloc[0]["_audit_Abs_Error"] == 35.0


def test_serialization_sidecar_logic(populated_summary, tmp_path):
    """
    Verify that _extract_preds_and_probs clears memory but preserves data for export.
    Crucial for handling the peak RAM in the audit.
    """
    # 1. Extract sidecar
    sidecar = populated_summary._extract_preds_and_probs()

    # Verify main results no longer hold large arrays
    assert populated_summary.results[0].preds_val is None

    # 2. Restore
    populated_summary._restore_preds_and_probs(sidecar)

    # Verify data is back
    assert populated_summary.results[0].preds_val is not None
    assert populated_summary.results[0].preds_val.iloc[0] == 20.0
