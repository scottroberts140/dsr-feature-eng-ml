"""Elastic Net regression model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type, cast

from dsr_utils import format_label_value_pairs
from sklearn.linear_model import ElasticNet as SklearnElasticNet

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
from dsr_feature_eng_ml.prefs_instance import prefs


@dataclass(frozen=True)
class ElasticNetParams(ModelParams):
    """
    Hyperparameters for Elastic Net regression models.

    Elastic Net is a linear regression model trained with L1 and L2 prior as
    regularizer. This combination allows for learning a sparse model where
    few of the weights are non-zero (like Lasso), while still maintaining
    the regularization properties of Ridge.
    """

    alpha: float = 1.0
    l1_ratio: float = 0.5
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: int = 1000
    tol: float = 1e-4
    warm_start: bool = False
    positive: bool = False
    selection: Literal["cyclic", "random"] = "cyclic"
    random_state: int | None = None
    task_type: TaskType = TaskType.REGRESSION
    scoring: ScoringMetric = ScoringMetric.R2

    def info(self) -> str:
        """Return a formatted summary of Elastic Net parameters."""
        data = [
            ("Alpha (Penalty)", f"{self.alpha}"),
            ("L1 Ratio (Mix)", f"{self.l1_ratio}"),
            ("Fit Intercept", f"{self.fit_intercept}"),
            ("Max Iterations", f"{self.max_iter}"),
            ("Tolerance", f"{self.tol}"),
            ("Selection", f"{self.selection}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Generate a standard hyperparameter search grid for Elastic Net.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns a compact grid focusing on moderate alpha and
            l1_ratio values. If False, explores a wider range of regularization
            strengths and mixing ratios.

        Returns
        -------
        dict[str, list[Any]]
            A parameter grid mapping 'alpha' and 'l1_ratio' to candidate values.
        """
        if narrow:
            return {"alpha": [0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]}

        return {
            "alpha": [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        }


class ElasticNetRegression(ModelSpecification[ElasticNetParams, SklearnElasticNet]):
    """
    Elastic Net regression model specification.

    This class handles the lifecycle of an Elastic Net model, including
    regularization path optimization and performance auditing for numerical targets.
    """

    params_class = ElasticNetParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[ElasticNetParams] = None,
        task_type: TaskType = TaskType.REGRESSION,
        scoring: ScoringMetric = ScoringMetric.R2,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the Elastic Net model specification."""
        if params is None:
            params = ElasticNetParams(
                task_type=task_type, random_state=1, scoring=scoring
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
            optimization_strategy=optimization_strategy,
        )

        self._model_type = ModelType.ELASTIC_NET
        self.estimator = self.create_estimator()

    @property
    def task_type(self) -> TaskType:
        """The regression task type for this model."""
        return self._task_type

    @property
    def model_type(self) -> ModelType:
        """The Elastic Net model type identifier."""
        return self._model_type

    @property
    def scoring(self) -> ScoringMetric:
        """The scoring metric used for optimization."""
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_dials(self) -> ElasticNetParams:
        """The hyperparameter container for the current model."""
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: ElasticNetParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> Type[SklearnElasticNet]:
        """Return the Scikit-Learn ElasticNet class."""
        return SklearnElasticNet

    def create_estimator(
        self, parameters: Optional[ElasticNetParams] = None
    ) -> SklearnElasticNet:
        """
        Instantiate a raw Scikit-Learn ElasticNet estimator.

        Parameters
        ----------
        parameters : ElasticNetParams, optional
            Parameter override. If None, uses instance dials.
        """
        p = parameters or self.model_dials

        return SklearnElasticNet(
            alpha=p.alpha,
            l1_ratio=p.l1_ratio,
            fit_intercept=p.fit_intercept,
            copy_X=p.copy_X,
            max_iter=p.max_iter,
            tol=p.tol,
            warm_start=p.warm_start,
            positive=p.positive,
            selection=cast(Literal["cyclic", "random"], p.selection),
            random_state=p.random_state,
        )
