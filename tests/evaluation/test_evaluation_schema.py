"""
Tests for dsr_feature_eng_ml.evaluation module.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelGeneralization,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.evaluation.schema import (
    DatasetFormatters,
    DataSplits,
    FeatureMetadata,
    ModelAuditorConfig,
    ModelConfiguration,
    ModelFeatureImportance,
)
from dsr_feature_eng_ml.models.lasso_regression import LassoRegression
from dsr_feature_eng_ml.models.random_forest import (
    RandomForestRegressorModel,
    RandomForestRegressorParams,
)
from dsr_utils.formatting import (
    BoolFormat,
    CurrencyFormat,
    DateTimeFormat,
    FloatFormat,
    IntegerFormat,
    StringFormat,
)
from sklearn.preprocessing import StandardScaler


def test_dataset_formatters_default_mapping():
    """
    Verify that dtypes map to the expected formatter instances by default.
    Ensures visual consistency across audit tables.
    """
    formatters = DatasetFormatters()

    # 1. Test Numeric Mappings
    assert isinstance(formatters.fmt_for_dtype(np.dtype("float64")), FloatFormat)
    assert isinstance(formatters.fmt_for_dtype(np.dtype("int64")), IntegerFormat)

    # 2. Test Boolean Mapping
    assert isinstance(formatters.fmt_for_dtype(np.dtype("bool")), BoolFormat)

    # 3. Test Temporal Mappings using pandas API
    assert isinstance(
        formatters.fmt_for_dtype(pd.Series([pd.Timestamp("2026-01-01")]).dtype),
        DateTimeFormat,
    )
    assert isinstance(
        formatters.fmt_for_dtype(pd.Series([pd.Timedelta(days=1)]).dtype),
        DateTimeFormat,
    )

    # 4. Test Categorical and Object Mappings
    assert isinstance(formatters.fmt_for_dtype(pd.CategoricalDtype()), StringFormat)
    assert isinstance(formatters.fmt_for_dtype(np.dtype("O")), StringFormat)


def test_dataset_formatters_setter_logic():
    """
    Verify that custom formatters can be injected (e.g., using Currency for fares).
    """
    formatters = DatasetFormatters()
    # Your audit uses Currency for fare_amount
    fare_formatter = CurrencyFormat(width=8, precision=2)
    formatters.dtype_float = fare_formatter

    resolved_fmt = formatters.fmt_for_dtype(np.dtype("float64"))
    assert resolved_fmt == fare_formatter
    # Ensure precision matches the '$ -83.50' output on Page 9
    assert resolved_fmt.format_value(-83.5) == "$  -83.50"


def test_feature_metadata_initialization():
    """Verify manual initialization of feature metadata."""
    fmt = CurrencyFormat()
    fm = FeatureMetadata(
        name="fare_amount",
        id="F08",
        position=7,
        short_name="Fare",
        formatter=fmt,
        is_used_in_fit=True,
    )

    assert fm.name == "fare_amount"
    assert fm.id == "F08"
    assert fm.short_name == "Fare"
    assert fm.is_used_in_fit is True
    assert isinstance(fm.formatter, CurrencyFormat)


def test_from_df_automatic_generation(mini_taxi_df):
    """
    Verify metadata generation from a DataFrame (Feature List).

    Data columns (total 20 columns):
    #   Column                 Dtype
    ---  ------                 -----
    0   VendorID               int32
    1   tpep_pickup_datetime   datetime64[us]
    2   tpep_dropoff_datetime  datetime64[us]
    3   passenger_count        float64
    4   trip_distance          float64
    5   RatecodeID             float64
    6   store_and_fwd_flag     str
    7   PULocationID           int32
    8   DOLocationID           int32
    9   payment_type           int64
    10  fare_amount            float64
    11  extra                  float64
    12  mta_tax                float64
    13  tip_amount             float64
    14  tolls_amount           float64
    15  improvement_surcharge  float64
    16  total_amount           float64
    17  congestion_surcharge   float64
    18  Airport_fee            float64
    19  cbd_congestion_fee     float64
    """
    exclude = {"tpep_pickup_datetime", "tpep_dropoff_datetime"}
    parents = {
        "fare_amount": "total_amount",
        "tip_amount": "total_amount",
    }
    shorts = {"trip_distance": "Distance"}

    fm_dict = FeatureMetadata.from_df(
        df=mini_taxi_df,
        exclude_from_fit=exclude,
        feature_parent=parents,
        short_names=shorts,
    )

    # 1. Check ID Generation (e.g., F05 for Distance)
    assert fm_dict["trip_distance"].id == "F05"
    assert fm_dict["trip_distance"].short_name == "Distance"

    # 2. Check Exclusion Logic
    assert fm_dict["tpep_dropoff_datetime"].is_used_in_fit is False
    assert fm_dict["passenger_count"].is_used_in_fit is True

    # 3. Check Parent Relationship
    assert fm_dict["fare_amount"].parent_name == "total_amount"
    assert fm_dict["VendorID"].parent_name is None


def test_feature_filtering_logic():
    """Verify list_to_set correctly filters features for model training."""
    f1 = FeatureMetadata("A", "F01", 0, is_used_in_fit=True)
    f2 = FeatureMetadata("B", "F02", 1, is_used_in_fit=False)
    f3 = FeatureMetadata("Target", "F03", 2, is_used_in_fit=True)

    # Should exclude B (is_used_in_fit=False) and the target column
    filtered_set = FeatureMetadata.list_to_set([f1, f2, f3], target_column="Target")

    assert len(filtered_set) == 1
    assert f1 in filtered_set
    assert f3 not in filtered_set


@pytest.fixture
def mock_feature_set():
    """Create a set of feature metadata mimicking the Taxi Audit."""
    return {
        FeatureMetadata(name="trip_distance", id="F05", position=2),
        FeatureMetadata(name="payment_type", id="F10", position=6),
        FeatureMetadata(name="RatecodeID", id="F06", position=3),
        FeatureMetadata(name="DOLocationID", id="F09", position=5),
    }


def test_importance_calculation_and_sorting(mock_feature_set):
    """Verify that importances are correctly sorted and cumulated."""
    # Simulate weights where trip_distance is dominant
    importances = np.array([0.75, 0.10, 0.10, 0.05])
    mfi = ModelFeatureImportance(mock_feature_set, importances)

    # 1. Verify Sorting (Highest Importance First)
    assert mfi.feature_importances.iloc[0]["importance"] == 0.75

    # 2. Verify Cumulative Calculation
    # First + Second feature (0.75 + 0.10) should be 0.85
    assert mfi.feature_importances.iloc[1]["cumulative_importance"] == 0.85


def test_threshold_index_identification(mock_feature_set):
    """Verify the 80% and 95% threshold logic used in charts."""
    # Distribution matching the audit cumulative curve
    importances = np.array([0.75, 0.15, 0.06, 0.04])
    mfi = ModelFeatureImportance(mock_feature_set, importances)
    mfi.calc_threshold_indices()

    # 80% Threshold: Feature 1 (0.75) < 0.8, Feature 2 (0.75+0.15=0.9) > 0.8
    assert mfi.threshold_80_idx == 2

    # 95% Threshold: Feature 3 (0.9 + 0.06 = 0.96) pushes it over 0.95
    # The logic adds +1 for the next feature boundary or clamps to total
    assert mfi.threshold_95_idx <= len(mock_feature_set)


def test_empty_importance_handling():
    """Ensure the 'empty' factory works for initializations."""
    empty_mfi = ModelFeatureImportance.empty()
    assert len(empty_mfi.features) == 0
    assert empty_mfi.threshold_80_idx == 0


def test_datasplits_factory_and_leakage(mini_taxi_df):
    """
    Verify that DataSplits correctly segments data and fits the scaler
    only on the training set to prevent data leakage.
    """
    target = "fare_amount"
    features = [c for c in mini_taxi_df.columns if c != target]

    splits = DataSplits.from_data_source(
        src=mini_taxi_df,
        features_to_include=features,
        target_column=target,
        test_size=0.2,
        valid_size=0.2,
        original_row_count=len(mini_taxi_df),
        random_state=42,
        scale_features=True,
    )

    # 1. Verify split proportions
    # Total 1000: Test=200, Main=800. Main Val=160, Main Train=640
    assert len(splits.train_features) == 640
    assert len(splits.val_features) == 160
    assert len(splits.test_features) == 200

    # 2. Leakage Check: Scaler must be fitted ONLY on training data
    assert isinstance(splits.scaler, StandardScaler)
    # Check that the scaler's mean_ is roughly the training mean, not the global mean
    assert splits.scaler.n_samples_seen_ == 640


def test_inverse_transform_logic(mini_taxi_df):
    """Verify that scaled features can be reverted to original units for reporting."""
    target = "fare_amount"
    features = ["trip_distance"]  # Select a known numeric feature

    splits = DataSplits.from_data_source(
        src=mini_taxi_df,
        features_to_include=features,
        target_column=target,
        test_size=0.2,
        valid_size=0.2,
        original_row_count=1000,
        random_state=42,
        scale_features=True,
    )

    scaled_val = splits.train_features.iloc[0]["trip_distance"]
    # The value in train_features should be z-score scaled (likely near 0)
    assert -5.0 < scaled_val < 5.0

    # Revert to original units
    df_inv = splits.inverse_transform_df(splits.train_features)
    original_val = df_inv.iloc[0]["trip_distance"]
    # Verify it matches a realistic taxi trip distance (likely > 0)
    assert original_val >= 0


def test_resampling_strategies(mini_taxi_df):
    """Verify that upsampling and downsampling produce balanced classes."""
    # Create a dummy binary target for classification testing
    df = mini_taxi_df.copy()
    df["binary_target"] = (df["fare_amount"] > 20).astype(int)

    splits = DataSplits.from_data_source(
        src=df,
        features_to_include=["trip_distance"],
        target_column="binary_target",
        test_size=0.2,
        valid_size=0.2,
        original_row_count=1000,
        random_state=42,
    )

    # Perform oversampling
    balanced_feat, balanced_targ = splits.get_balanced_train_data(
        strategy=BalancingStrategy.OVERSAMPLED,
        feature_set=set(),  # Logic handles feature list extraction
        use_combined_data=False,
    )

    # Check that class counts are now equal
    counts = balanced_targ.value_counts()
    assert counts[0] == counts[1]


@pytest.fixture
def rf_config_from_audit():
    """Build a ModelConfiguration using the specific values from the Taxi Audit JSON."""
    params = RandomForestRegressorParams(
        n_estimators=50, max_depth=20, min_samples_leaf=5
    )

    return ModelConfiguration(
        id="01",
        model_type=ModelType.RANDOM_FOREST_REGRESSOR,
        task_type=TaskType.REGRESSION,
        balancing_strategy=BalancingStrategy.NONE,
        optimization_strategy=OptimizationStrategy.RANDOM_SEARCH,
        model_params=params,
        cv=5,
        scoring=ScoringMetric.R2,
        n_jobs=2,
        n_iter=5,
        # Performance metrics from JSON
        score_cv=0.710853,
        score_train=0.818314,
        score_val=0.814474,
        score_val_cleaned=0.894007,
        r2_train=0.818314,
        r2_val=0.814474,
        mae_train=2.8789,
        mae_val=2.7088,
        # Telemetry
        tuning_duration=5.845,
        fit_duration=0.310,
    )


def test_performance_gap_logic(rf_config_from_audit):
    """Verify that R2 and MAE gaps are calculated correctly."""
    # r2_gap = abs(0.818314 - 0.814474) = 0.00384
    assert round(rf_config_from_audit.r2_gap, 5) == 0.00384

    # mae_gap = 2.7088 - 2.8789 = -0.1701
    assert round(rf_config_from_audit.mae_gap, 4) == -0.1701

    # Validation of the generalization status based on prefs.acceptable_gap (0.02)
    # Since 0.00384 < 0.02, it should be WELL_FIT
    assert rf_config_from_audit.model_generalization == ModelGeneralization.WELL_FIT


def test_throughput_efficiency(rf_config_from_audit):
    """Verify the rows-per-second calculation seen in the Winning Model summary."""

    # Mock splits with 8000 rows
    class MockSplits:
        train_features = [None] * 6400
        val_features = [None] * 1600

    # Total duration = 5.845 + 0.310 = 6.155s
    # Throughput = 8000 / 6.155 = ~1299.75
    efficiency = rf_config_from_audit.efficiency(MockSplits())
    assert 1290 < efficiency < 1310


def test_top_feature_extraction(rf_config_from_audit):
    """Verify that the flattened feature dictionary matches report expectations."""
    from dsr_feature_eng_ml.evaluation.schema import (
        FeatureMetadata,
        ModelFeatureImportance,
    )

    # 1. Prepare the metadata and importance array
    f_set = {FeatureMetadata(name="trip_distance", id="F03", position=2)}
    importances_array = np.array([0.7550], dtype=np.float32)

    # 2. Create the importance analysis object
    analysis = ModelFeatureImportance(feature_set=f_set, importances=importances_array)

    # 3. Use replace() to bypass the FrozenInstanceError
    updated_config = replace(rf_config_from_audit, feature_analysis=analysis)

    # 4. Verify the extraction from the new instance
    top_feature_report = updated_config.get_top_features(n=1)
    assert top_feature_report["Top_Feature_1"] == "trip_distance"
    assert top_feature_report["Importance_1"] == 0.755


def test_auditor_config_from_dataset(mini_taxi_df):
    """
    Verify the factory method creates splits and instantiates the model list correctly.
    Matches the '6.16s' throughput logic for multiple models.
    """
    target = "fare_amount"

    config = ModelAuditorConfig.from_dataset(
        dataset=mini_taxi_df,
        original_row_count=len(mini_taxi_df),
        target_column=target,
        dataset_name="Taxi Test",
        cv=3,
        model_classes=[RandomForestRegressorModel, LassoRegression],
        balancing_strategies=[BalancingStrategy.NONE],
        task_type=TaskType.REGRESSION,
        random_state=42,
    )

    # 1. Verify split propagation
    assert isinstance(config.data_splits, DataSplits)
    assert config.data_splits.target_column == target

    # 2. Verify model instantiation
    assert len(config.models_to_run) == 2
    assert any(isinstance(m, RandomForestRegressorModel) for m in config.models_to_run)

    # 3. Verify parameter propagation (e.g., CV)
    for model in config.models_to_run:
        assert model.cv == 3


def test_n_jobs_propagation():
    """Verify that updating n_jobs at the config level updates all models."""
    # Create empty splits for manual instantiation
    empty_splits = DataSplits.empty()

    m1 = RandomForestRegressorModel(cv=5)
    m2 = LassoRegression(cv=5, balancing_strategy=BalancingStrategy.NONE)

    config = ModelAuditorConfig(
        data_splits=empty_splits, dataset_name="Job Test", models_to_run=[m1, m2]
    )

    # Default n_jobs is set to 3 in __post_init__
    assert config.n_jobs == 3
    assert m1.n_jobs == 3

    # Update through setter and check propagation
    config.n_jobs = 5
    assert m1.n_jobs == 5
    assert m2.n_jobs == 5


def test_risk_threshold_loading():
    """Ensure preferences are correctly loaded into the config defaults."""
    # This verifies the link to prefs.anomaly_threshold and others
    config = ModelAuditorConfig(
        data_splits=DataSplits.empty(), dataset_name="Threshold Test"
    )

    # Matches the 'Drift: 3.811%' logic on Page 6
    assert config.drift_threshold > 0.0
    assert config.anomaly_threshold > 0.0


def test_data_splits_skip_encoding_falls_back_for_non_numeric_columns(capsys):
    """Verify string skip_encoding columns fall back to one-hot encoding."""
    df = pd.DataFrame(
        {
            "city": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A"],
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    splits = DataSplits.from_data_source(
        src=df,
        features_to_include=["city", "num"],
        target_column="target",
        test_size=0.2,
        valid_size=0.25,
        original_row_count=len(df),
        random_state=42,
        scale_features=True,
        skip_encoding=["city"],
    )

    captured = capsys.readouterr()
    assert "skip_encoding fallback to one-hot" in captured.out
    assert any(col.startswith("city_") for col in splits.train_features.columns)
    assert not splits.train_features["num"].isna().all()
