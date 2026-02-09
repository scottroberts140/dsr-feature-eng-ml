"""Ridge regression model specification and parameter definitions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Type
from dsr_feature_eng_ml.enums import (
    ModelType,
    BalancingStrategy,
    ScoringMetric,
    TaskType,
    OptimizationStrategy,
)
from dsr_feature_eng_ml.models.model_specification import (
    ModelSpecification,
    ModelParams,
)
from dsr_feature_eng_ml.evaluation.schema import DataSplits
from dsr_feature_eng_ml.preferences import prefs
from dsr_utils import format_label_value_pairs
from sklearn.linear_model import Ridge as SklearnRidge


@dataclass(frozen=True)
class RidgeParams(ModelParams):
    """Hyperparameters for ridge regression models."""

    alpha: float = 1.0
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: Optional[int] = None
    tol: float = 1e-4
    solver: Literal[
        "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"
    ] = "auto"
    random_state: Optional[int] = None
    task_type: TaskType = TaskType.REGRESSION
    scoring: ScoringMetric = ScoringMetric.R2

    def info(self) -> str:
        data = [
            ("Alpha (Penalty)", f"{self.alpha}"),
            ("Solver", f"{self.solver}"),
            ("Fit Intercept", f"{self.fit_intercept}"),
            ("Tolerance", f"{self.tol}"),
            ("Task Type", f"{self.task_type}"),
            ("Scoring", f"{self.scoring}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list]:
        """Generate a standard hyperparameter search grid for Ridge Regression.

        Provides a small, logarithmic grid for quick tuning, or an expanded
        grid that spans multiple orders of magnitude.

        Args:
            narrow (bool, optional): If True (default), returns a compact grid
                for quick searches. If False, returns a wider grid.

        Returns:
            dict[str, list]: Parameter grid mapping ``alpha`` to candidate values.
                - narrow=True: alpha [0.1, 1.0, 10.0, 100.0]
                - narrow=False: alpha [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

        Example:
            >>> narrow_grid = RidgeParams.get_standard_search_grid()
            >>> expanded_grid = RidgeParams.get_standard_search_grid(narrow=False)
        """
        # Regularization strength is best explored logarithmically
        if narrow:
            return {"alpha": [0.1, 1.0, 10.0, 100.0]}

        # Expanded grid covers a much wider range of penalty strengths
        return {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}


class RidgeRegression(ModelSpecification[RidgeParams, SklearnRidge]):
    """Ridge regression model specification."""

    def get_estimator_class(
        self,
    ) -> Type[SklearnRidge]:
        return SklearnRidge

    params_class = RidgeParams

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric):
        self._scoring = value

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def model_dials(self) -> RidgeParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: RidgeParams) -> None:
        self._model_dials = value

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy,
        params: Optional[RidgeParams] = None,
        task_type: TaskType = TaskType.REGRESSION,
        scoring: ScoringMetric = ScoringMetric.R2,
        n_jobs: int = -1,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        # Use provided params or fall back to defaults
        # We ensure the data's random_state is used if not specified in params
        if params is None:
            params = RidgeParams(
                task_type=task_type,
                random_state=1,
            )

        self._model_dials = params
        self._task_type = TaskType.REGRESSION
        self._scoring = self.model_dials.scoring

        super().__init__(
            cv=cv,
            balancing_strategy=balancing_strategy,
            n_jobs=n_jobs,
            n_iter=n_iter,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap,
        )

        self._model_type = ModelType.RIDGE
        self.optimization_strategy = optimization_strategy
        self.estimator = self.create_estimator()

    def create_estimator(
        self, parameters: Optional[RidgeParams] = None
    ) -> SklearnRidge:
        params = parameters or self.model_dials

        return SklearnRidge(
            alpha=params.alpha,
            fit_intercept=params.fit_intercept,
            copy_X=params.copy_X,
            max_iter=params.max_iter,
            tol=params.tol,
            solver=params.solver,
            random_state=params.random_state,
        )
