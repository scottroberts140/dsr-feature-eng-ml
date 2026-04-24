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
