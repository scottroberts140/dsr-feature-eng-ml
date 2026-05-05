from dataclasses import dataclass
from typing import Any, Optional, Type, Union

import numpy as np
import pytest
from dsr_feature_eng_ml.enums import (
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.models.decision_tree import DecisionTreeRegressorModel
from dsr_feature_eng_ml.models.model_specification import (
    ModelParams,
    ModelSpecification,
)
from dsr_feature_eng_ml.prefs_instance import prefs

# --- MOCK IMPLEMENTATIONS FOR TESTING ---


@dataclass(frozen=True)
class MockParams(ModelParams):
    """
    Concrete implementation of ModelParams.
    Updated alpha to accept lists for search grid testing.
    """

    alpha: Union[float, list[float], tuple[float, ...]] = 1.0
    n_estimators: int = 10

    def info(self) -> str:
        return f"Mock(alpha={self.alpha})"

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        return {"alpha": [0.1, 0.5, 1.0]}


class MockModelSpec(ModelSpecification[MockParams, Any]):
    """Concrete implementation of ModelSpecification."""

    params_class = MockParams

    def __init__(self, **kwargs):
        # Default mandatory fields for the base constructor
        kwargs.setdefault("cv", 5)
        kwargs.setdefault("optimization_strategy", OptimizationStrategy.MANUAL)
        super().__init__(**kwargs)
        self._model_type = ModelType.UNKNOWN
        self._params = kwargs.get("params") or MockParams()

    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION

    @property
    def scoring(self) -> ScoringMetric:
        return ScoringMetric.R2

    @scoring.setter
    def scoring(self, value: ScoringMetric):
        pass

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def model_dials(self) -> MockParams:
        return self._params

    @model_dials.setter
    def model_dials(self, value: MockParams):
        self._params = value

    def get_estimator_class(self) -> Type[Any]:
        return object

    def create_estimator(self, parameters: Optional[MockParams] = None) -> Any:
        # Returns a mock with fit/predict to satisfy ScikitModel protocol
        class DummyEstimator:
            def fit(self, X, y, sample_weight=None):
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def get_params(self, deep=True):
                return {"alpha": 1.0}

        return DummyEstimator()


@dataclass(frozen=True)
class MockFeature:
    name: str


class MockDataSplits:
    def __init__(self):
        self.train_features = __import__("pandas").DataFrame(
            {"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]}
        )
        self.train_target = __import__("pandas").Series([0.1, 0.2, 0.3])

    def get_balanced_train_data(self, strategy, feature_set, use_combined_data=False):
        feature_list = [f.name for f in feature_set]
        return self.train_features[feature_list], self.train_target

    def get_train_weights(self, balancing_strategy, is_regression):
        return None


class VerboseFitModelSpec(MockModelSpec):
    def create_estimator(self, parameters: Optional[MockParams] = None) -> Any:
        class VerboseEstimator:
            def __init__(self):
                self.fit_kwargs = {}

            def fit(self, X, y, sample_weight=None, verbose=0):
                self.fit_kwargs = {
                    "sample_weight": sample_weight,
                    "verbose": verbose,
                }
                return self

            def predict(self, X):
                return np.zeros(len(X))

            def get_params(self, deep=True):
                return {"alpha": 1.0}

        return VerboseEstimator()


# --- TEST SUITE ---


def test_base_params_serialization():
    """Verify ModelParams converts to dict and handles Enum naming."""
    params = MockParams(optimization_strategy=OptimizationStrategy.RANDOM_SEARCH)
    p_dict = params.to_dict()

    # 1. Check standardization of Enum to string name
    assert p_dict["optimization_strategy"] == "RANDOM_SEARCH"
    # 2. Verify basic field retention
    assert p_dict["alpha"] == 1.0


def test_search_iteration_calculation():
    """Verify n_iter logic for Grid vs Random search."""
    # 1. Manual strategy: scalar alpha, returns 1
    params_manual = MockParams(
        optimization_strategy=OptimizationStrategy.MANUAL, alpha=1.0
    )
    assert params_manual.num_candidates == 1

    # 2. Grid search: list for alpha, returns 3
    params_grid = MockParams(
        optimization_strategy=OptimizationStrategy.GRID_SEARCH, alpha=[0.1, 0.5, 1.0]
    )
    # The logic in ModelParams uses math.prod on lengths of lists/tuples
    assert params_grid.num_candidates == 3


def test_model_specification_metric_validation():
    """Verify constructor raises error on task/metric mismatch."""
    with pytest.raises(ValueError, match="Invalid metric"):
        # F1 is invalid for a Regression task_type
        DecisionTreeRegressorModel(cv=5, scoring=ScoringMetric.F1)


def test_feature_importance_extraction_logic():
    """Verify extraction of coefficients and feature importances."""
    spec = MockModelSpec()

    # Mock a fitted estimator with coefficients
    class LinearEstimator:
        def fit(self, X, y):
            pass

        def predict(self, X):
            pass

        def get_params(self, deep=True):
            return {}

        coef_ = np.array([-0.5, 0.8])

    spec.estimator = LinearEstimator()
    importances = spec.feature_importances

    # 1. Explicitly assert not None to satisfy type checker
    assert importances is not None, (
        "Feature importances should not be None for fitted linear model"
    )

    # 2. Perform the equality check
    # Coefficients are returned as absolute values
    np.testing.assert_array_almost_equal(importances, np.array([0.5, 0.8]))


def test_fit_forwards_verbose_when_supported(monkeypatch):
    """fit() should pass fit_verbose when estimator.fit accepts verbose."""
    old_fit_verbose = prefs.fit_verbose
    monkeypatch.setattr(prefs, "fit_verbose", 2)

    spec = VerboseFitModelSpec()
    splits = MockDataSplits()
    features = {MockFeature("x1"), MockFeature("x2")}

    spec.fit(data_splits=splits, features_to_fit_set=features)

    assert spec.estimator is not None
    assert spec.estimator.fit_kwargs["verbose"] == 2

    monkeypatch.setattr(prefs, "fit_verbose", old_fit_verbose)


def test_fit_skips_verbose_when_not_supported():
    """fit() should not pass verbose when estimator.fit lacks that parameter."""
    spec = MockModelSpec()
    splits = MockDataSplits()
    features = {MockFeature("x1"), MockFeature("x2")}

    # Should complete without raising TypeError from an unexpected verbose kwarg
    mem_used, mem_peak = spec.fit(data_splits=splits, features_to_fit_set=features)

    assert mem_used >= 0
    assert mem_peak >= 0
