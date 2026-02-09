"""Elastic Net regression model specification and parameter definitions."""

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
from sklearn.linear_model import ElasticNet as SklearnElasticNet


@dataclass(frozen=True)
class ElasticNetParams(ModelParams):
    """Hyperparameters for Elastic Net regression models."""

    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    selection: Literal["cyclic", "random"] = "cyclic"
    random_state: Optional[int] = None
    task_type: TaskType = TaskType.REGRESSION
    scoring: ScoringMetric = ScoringMetric.R2

    def info(self) -> str:
        data = [
            ("Alpha (Penalty)", f"{self.alpha}"),
            ("L1 Ratio (Mix)", f"{self.l1_ratio}"),
            ("Fit Intercept", f"{self.fit_intercept}"),
            ("Max Iterations", f"{self.max_iter}"),
            ("Tolerance", f"{self.tol}"),
            ("Selection", f"{self.selection}"),
            ("Task Type", f"{self.task_type}"),
            ("Scoring", f"{self.scoring}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list]:
        """Generate a standard hyperparameter search grid for Elastic Net.

        The narrow grid focuses on moderate alpha and l1_ratio values, while the
        expanded grid explores a wider range of regularization strengths and
        mixing ratios.

        Args:
            narrow (bool, optional): If True (default), returns a compact grid.
                If False, returns an expanded grid.

        Returns:
            dict[str, list]: Parameter grid mapping ``alpha`` and ``l1_ratio`` to
            candidate values.

        Example:
            >>> narrow_grid = ElasticNetParams.get_standard_search_grid()
            >>> expanded_grid = ElasticNetParams.get_standard_search_grid(narrow=False)
        """
        if narrow:
            return {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]}
        return {
            "alpha": [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        }


class ElasticNetRegression(ModelSpecification[ElasticNetParams, SklearnElasticNet]):
    """Elastic Net regression model specification."""

    def get_estimator_class(
        self,
    ) -> Type[SklearnElasticNet]:
        return SklearnElasticNet

    params_class = ElasticNetParams

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric):
        self._scoring = value

    @property
    def model_dials(self) -> ElasticNetParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: ElasticNetParams) -> None:
        self._model_dials = value

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy,
        params: Optional[ElasticNetParams] = None,
        task_type: TaskType = TaskType.REGRESSION,
        scoring: ScoringMetric = ScoringMetric.R2,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        # Use provided params or fall back to defaults
        # We ensure the data's random_state is used if not specified in params
        if params is None:
            params = ElasticNetParams(
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

        self._model_type = ModelType.ELASTIC_NET
        self.optimization_strategy = optimization_strategy
        self.estimator = self.create_estimator()

    def create_estimator(
        self, parameters: Optional[ElasticNetParams] = None
    ) -> SklearnElasticNet:
        params = parameters or self.model_dials
        return SklearnElasticNet(
            alpha=params.alpha,
            l1_ratio=params.l1_ratio,
            fit_intercept=params.fit_intercept,
            copy_X=params.copy_X,
            max_iter=params.max_iter,
            tol=params.tol,
            warm_start=params.warm_start,
            positive=params.positive,
            selection=params.selection,
            random_state=params.random_state,
        )
