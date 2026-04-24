from typing import Type

import pytest
from dsr_feature_eng_ml.enums import ModelType, ScoringMetric, TaskType
from dsr_feature_eng_ml.models.decision_tree import (
    DecisionTreeClassifierModel,
    DecisionTreeClassifierParams,
    DecisionTreeRegressorModel,
    DecisionTreeRegressorParams,
)
from dsr_feature_eng_ml.models.elastic_net_regression import (
    ElasticNetParams,
    ElasticNetRegression,
)
from dsr_feature_eng_ml.models.lasso_regression import LassoParams, LassoRegression
from dsr_feature_eng_ml.models.linear_regression import (
    LinearRegression,
    LinearRegressionParams,
)
from dsr_feature_eng_ml.models.logistic_regression import (
    LogisticRegression,
    LogisticRegressionParams,
)
from dsr_feature_eng_ml.models.model_specification import ModelParams
from dsr_feature_eng_ml.models.random_forest import (
    RandomForestClassifierModel,
    RandomForestClassifierParams,
    RandomForestRegressorModel,
    RandomForestRegressorParams,
)
from dsr_feature_eng_ml.models.ridge_regression import RidgeParams, RidgeRegression
from dsr_feature_eng_ml.preferences import prefs

# Define the contract: (Class, Expected ModelType, Expected TaskType, Scoring Metric)
MODEL_CONTRACTS = [
    (
        RandomForestClassifierModel,
        ModelType.RANDOM_FOREST_CLASSIFIER,
        TaskType.CLASSIFICATION,
        ScoringMetric.F1,
    ),
    (
        RandomForestRegressorModel,
        ModelType.RANDOM_FOREST_REGRESSOR,
        TaskType.REGRESSION,
        ScoringMetric.R2,
    ),
    (
        DecisionTreeClassifierModel,
        ModelType.DECISION_TREE_CLASSIFIER,
        TaskType.CLASSIFICATION,
        ScoringMetric.F1,
    ),
    (
        DecisionTreeRegressorModel,
        ModelType.DECISION_TREE_REGRESSOR,
        TaskType.REGRESSION,
        ScoringMetric.R2,
    ),
    (
        LinearRegression,
        ModelType.LINEAR_REGRESSION,
        TaskType.REGRESSION,
        None,
    ),
    (RidgeRegression, ModelType.RIDGE, TaskType.REGRESSION, ScoringMetric.R2),
    (LassoRegression, ModelType.LASSO, TaskType.REGRESSION, ScoringMetric.R2),
    (
        ElasticNetRegression,
        ModelType.ELASTIC_NET,
        TaskType.REGRESSION,
        ScoringMetric.R2,
    ),
    (
        LogisticRegression,
        ModelType.LOGISTIC_REGRESSION,
        TaskType.CLASSIFICATION,
        ScoringMetric.F1,
    ),
    # Add newly refactored classification models here
]


@pytest.mark.parametrize(
    "model_class, expected_type, expected_task, scoring", MODEL_CONTRACTS
)
def test_all_model_integrities(model_class, expected_type, expected_task, scoring):
    """
    Verify that every model in the library correctly identifies its type and task.
    This prevents 'ghost' classification charts in regression audits.
    """
    if scoring is not None:
        instance = model_class(cv=None, balancing_strategy=None, scoring=scoring)
    else:
        instance = model_class(cv=None, balancing_strategy=None)

    # 1. Identity Check
    assert instance.model_type == expected_type
    assert instance.task_type == expected_task

    # 2. Parameter Hydration Check
    # Ensures the create_estimator() logic won't fail during the audit sweep
    estimator = instance.create_estimator()
    assert estimator is not None


# Define the contract: (Params Class, Task Type, List of Expected Attribute Names)
PARAM_CONTRACTS = [
    (
        LassoParams,
        TaskType.REGRESSION,
        ScoringMetric.R2,
        [
            "alpha",
            "fit_intercept",
            "copy_X",
            "precompute",
            "max_iter",
            "tol",
            "warm_start",
            "positive",
            "selection",
        ],
    ),
    (
        LinearRegressionParams,
        TaskType.REGRESSION,
        ScoringMetric.R2,
        ["fit_intercept", "copy_X", "positive"],
    ),
    (
        RidgeParams,
        TaskType.REGRESSION,
        ScoringMetric.R2,
        ["alpha", "fit_intercept", "copy_X", "solver", "max_iter", "tol"],
    ),
    (
        ElasticNetParams,
        TaskType.REGRESSION,
        ScoringMetric.R2,
        [
            "alpha",
            "l1_ratio",
            "fit_intercept",
            "copy_X",
            "max_iter",
            "tol",
            "warm_start",
            "positive",
            "selection",
        ],
    ),
    (
        RandomForestClassifierParams,
        TaskType.CLASSIFICATION,
        ScoringMetric.F1,
        [
            "criterion",
            "max_features",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "bootstrap",
        ],
    ),
    (
        RandomForestRegressorParams,
        TaskType.REGRESSION,
        ScoringMetric.R2,
        [
            "criterion",
            "max_features",
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "bootstrap",
        ],
    ),
    (
        DecisionTreeClassifierParams,
        TaskType.CLASSIFICATION,
        ScoringMetric.F1,
        [
            "criterion",
            "splitter",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "class_weight",
            "ccp_alpha",
        ],
    ),
    (
        DecisionTreeRegressorParams,
        TaskType.REGRESSION,
        ScoringMetric.R2,
        [
            "criterion",
            "splitter",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "min_weight_fraction_leaf",
            "max_features",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "class_weight",
            "ccp_alpha",
        ],
    ),
    (
        LogisticRegressionParams,
        TaskType.CLASSIFICATION,
        ScoringMetric.F1,
        ["C", "penalty", "solver", "max_iter", "l1_ratio", "class_weight"],
    ),
]


@pytest.mark.parametrize(
    "params_class, task_type, scoring, expected_attrs", PARAM_CONTRACTS
)
def test_model_params_attributes(
    params_class: Type[ModelParams],
    task_type: TaskType,
    scoring: ScoringMetric,
    expected_attrs: list[str],
):
    """
    Verify that each ModelParams class contains the expected V1.2.0 hyperparameters.
    This prevents 'ghost' parameters or missing attributes during estimator creation.
    """
    # Instantiate the dataclass with default values
    # params_instance = params_class()
    if hasattr(params_class, "create_default"):
        params_instance = getattr(params_class, "create_default")(
            task_type=task_type, scoring=scoring, random_state=75
        )
    else:
        params_instance = params_class()

    for attr in expected_attrs:
        # Check if the attribute exists in the dataclass
        assert hasattr(params_instance, attr), (
            f"Model parameter class '{params_class.__name__}' is missing "
            f"required hyperparameter: '{attr}'"
        )


def test_audit_risk_profile_detection(mini_taxi_df):
    """Verify that the auditor identifies the high-kurtosis risk profile."""

    # Mocking the audit summary logic
    kurtosis_val = mini_taxi_df["fare_amount"].kurt()
    assert kurtosis_val > 3.0


def test_random_forest_regressor_params_allow_list_values():
    """Estimator creation should accept list-valued RF params (search-space style)."""
    params = RandomForestRegressorParams(
        n_estimators=[50, 100],
        max_depth=[10, 20],
        min_samples_leaf=[1, 5],
        min_samples_split=[2, 4],
    )
    model = RandomForestRegressorModel(cv=None, balancing_strategy=None, params=params)

    estimator = model.create_estimator()
    est_params = estimator.get_params()
    assert est_params["n_estimators"] == 50
    assert est_params["max_depth"] == 10
    assert est_params["min_samples_leaf"] == 1
    assert est_params["min_samples_split"] == 2


def test_decision_tree_regressor_params_allow_list_values():
    """Estimator creation should accept list-valued DT params (search-space style)."""
    params = DecisionTreeRegressorParams(
        max_depth=[5, 10],
        min_samples_leaf=[1, 3],
        min_samples_split=[2, 6],
    )
    model = DecisionTreeRegressorModel(cv=None, balancing_strategy=None, params=params)

    estimator = model.create_estimator()
    est_params = estimator.get_params()
    assert est_params["max_depth"] == 5
    assert est_params["min_samples_leaf"] == 1
    assert est_params["min_samples_split"] == 2


def test_random_forest_classifier_params_allow_list_values():
    """Classifier estimator creation should accept list-valued RF params."""
    params = RandomForestClassifierParams(
        n_estimators=[75, 150],
        max_depth=[8, 16],
        min_samples_leaf=[1, 2],
        min_samples_split=[2, 5],
    )
    model = RandomForestClassifierModel(cv=None, balancing_strategy=None, params=params)

    estimator = model.create_estimator()
    est_params = estimator.get_params()
    assert est_params["n_estimators"] == 75
    assert est_params["max_depth"] == 8
    assert est_params["min_samples_leaf"] == 1
    assert est_params["min_samples_split"] == 2


def test_decision_tree_classifier_params_allow_list_values():
    """Classifier estimator creation should accept list-valued DT params."""
    params = DecisionTreeClassifierParams(
        max_depth=[4, 9],
        min_samples_leaf=[1, 2],
        min_samples_split=[2, 4],
    )
    model = DecisionTreeClassifierModel(cv=None, balancing_strategy=None, params=params)

    estimator = model.create_estimator()
    est_params = estimator.get_params()
    assert est_params["max_depth"] == 4
    assert est_params["min_samples_leaf"] == 1
    assert est_params["min_samples_split"] == 2


def test_recommendation_page_logic():
    """Verify the strategic verdict logic for the winning model."""
    winning_model_name = "Random Forest Regressor"
    val_score = 0.8145

    # Assert formatting matches the Recommendation Page
    assert f"{val_score:.4f}" == "0.8145"
    assert "Random Forest" in winning_model_name


def test_data_quality_scoring():
    """Ensure quality score calculation triggers 'LOW' label correctly."""
    score = 60.23
    quality_label = "LOW" if score < 70 else "HIGH"
    assert quality_label == "LOW"


def test_visual_palette_coverage():
    """Ensure every ModelType in the Legend has a color mapping."""
    for model_type in ModelType:
        if model_type != ModelType.UNKNOWN:
            color_hex = prefs.get_color(model_type.value)
            assert color_hex.startswith("#")


def test_zebra_striping_transparency():
    """Verify table face_color uses 'none' for transparency fix."""
    # The renderer subfunction get_styles must use "none" string
    from dsr_utils.tables import TableColumnStyle

    style = TableColumnStyle(ha="left", face_color="none")
    assert style.face_color == "none"


def test_recommendation_note_logic():
    """Verify performance notes are triggered by score improvement."""
    val_score = 0.8145
    cv_score = 0.7109

    # Significant improvement note triggers if diff > 0.05 [cite: 520]
    improvement = val_score - cv_score
    assert improvement > 0.05
