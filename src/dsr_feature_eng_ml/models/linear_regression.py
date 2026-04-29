"""Linear regression model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass

from dsr_utils import format_label_value_pairs
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.models.model_specification import (
    RegressionModelParams,
    RegressionModelSpecification,
)
from dsr_feature_eng_ml.prefs_instance import prefs


@dataclass(frozen=True)
class LinearRegressionParams(RegressionModelParams):
    """Hyperparameters for linear regression models."""

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: int | None = None
    positive: bool = False
    random_state: int | None = None

    def info(self) -> str:
        data = [
            ("Fit Intercept", f"{self.fit_intercept}"),
            ("Positive Only", f"{self.positive}"),
            ("Task Type", f"{self.task_type}"),
            ("Scoring", f"{self.scoring}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list]:
        # Linear Regression usually isn't "tuned" in the same way,
        # but we provide the grid for interface consistency.
        return {"fit_intercept": [True, False], "positive": [True, False]}


class LinearRegression(
    RegressionModelSpecification[LinearRegressionParams, SklearnLinearRegression]
):
    """Linear regression model specification."""

    def get_estimator_class(
        self,
    ) -> type[SklearnLinearRegression]:
        return SklearnLinearRegression

    params_class = LinearRegressionParams

    @property
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_type(self) -> ModelType:
        return ModelType.LINEAR_REGRESSION

    @property
    def model_dials(self) -> LinearRegressionParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: LinearRegressionParams) -> None:
        self._model_dials = value

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy,
        params: LinearRegressionParams | None = None,
        task_type: TaskType = TaskType.REGRESSION,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        # Use provided params or fall back to defaults
        # We ensure the data's random_state is used if not specified in params
        if params is None:
            params = LinearRegressionParams(
                task_type=task_type,
                random_state=1,
            )

        self._model_dials = params
        self._scoring = self.model_dials.scoring

        super().__init__(
            cv=cv,
            balancing_strategy=balancing_strategy,
            n_jobs=n_jobs,
            n_iter=n_iter,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap,
        )

        self.optimization_strategy = optimization_strategy
        self.estimator = self.create_estimator()

    def create_estimator(
        self, parameters: LinearRegressionParams | None = None
    ) -> SklearnLinearRegression:
        params = parameters or self.model_dials

        return SklearnLinearRegression(
            fit_intercept=params.fit_intercept,
            copy_X=params.copy_X,
            n_jobs=params.n_jobs,
            positive=params.positive,
        )
