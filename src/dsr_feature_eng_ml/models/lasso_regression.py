"""Lasso regression model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

from dsr_utils import format_label_value_pairs
from sklearn.linear_model import Lasso as SklearnLasso

from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.models.model_specification import (
    ModelParams,
    ModelSpecification,
)
from dsr_feature_eng_ml.preferences import prefs


@dataclass(frozen=True)
class LassoParams(ModelParams):
    """Hyperparameters for Lasso (L1) regression models."""

    alpha: float = 1.0
    fit_intercept: bool = True
    copy_X: bool = True
    precompute: bool = False
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    selection: Literal["cyclic", "random"] = "cyclic"
    random_state: Optional[int] = 1
    task_type: TaskType = TaskType.REGRESSION
    scoring: ScoringMetric = ScoringMetric.R2

    def info(self) -> str:
        """Format parameters for diagnostic display."""
        data = [
            ("Alpha (Penalty)", f"{self.alpha}"),
            ("Fit Intercept", f"{self.fit_intercept}"),
            ("Max Iterations", f"{self.max_iter}"),
            ("Tolerance", f"{self.tol}"),
            ("Selection", f"{self.selection}"),
            ("Positive Only", f"{self.positive}"),
            ("Task Type", f"{self.task_type.value}"),
            ("Scoring", f"{self.scoring.value}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Generate a hyperparameter search grid for Lasso.

        Lasso typically requires finer alpha values to find the sparse
        feature set without zeroing out all coefficients.
        """
        if narrow:
            return {"alpha": [0.01, 0.1, 1.0, 10.0]}
        return {"alpha": [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]}


class LassoRegression(ModelSpecification[LassoParams, SklearnLasso]):
    """Lasso regression model specification implementation."""

    params_class = LassoParams

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy,
        params: Optional[LassoParams] = None,
        task_type: TaskType = TaskType.REGRESSION,
        scoring: ScoringMetric = ScoringMetric.R2,
        n_jobs: int = -1,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        # 1. Parameter Initialization
        if params is None:
            params = LassoParams(task_type=task_type, scoring=scoring)

        self._model_dials = params
        self._task_type = TaskType.REGRESSION  # Explicitly enforced for Lasso
        self._scoring = params.scoring
        self._model_type = ModelType.LASSO

        # 2. Base Class Orchestration
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

    def get_estimator_class(self) -> Type[SklearnLasso]:
        """Return the underlying scikit-learn Lasso class."""
        return SklearnLasso

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
    def model_dials(self) -> LassoParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: LassoParams) -> None:
        self._model_dials = value

    def create_estimator(
        self, parameters: Optional[LassoParams] = None
    ) -> SklearnLasso:
        """Hydrate the scikit-learn estimator with Lasso-specific dials."""
        p = parameters or self.model_dials
        return SklearnLasso(
            alpha=p.alpha,
            fit_intercept=p.fit_intercept,
            copy_X=p.copy_X,
            precompute=p.precompute,
            max_iter=p.max_iter,
            tol=p.tol,
            warm_start=p.warm_start,
            positive=p.positive,
            selection=p.selection,
            random_state=p.random_state,
        )
