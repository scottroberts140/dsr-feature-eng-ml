"""Logistic regression model specification and parameter definitions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional, Literal, Type
import numpy as np
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
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


@dataclass(frozen=True)
class LogisticRegressionParams(ModelParams):
    """Hyperparameters for logistic regression models."""

    penalty: Literal["l1", "l2", "elasticnet", None] = "l2"
    C: float = 1.0
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs"
    max_iter: int = 100
    class_weight: Optional[Union[dict, str]] = None
    l1_ratio: Optional[float] = None  # Only used if penalty='elasticnet'
    random_state: Optional[int] = None
    task_type: TaskType = TaskType.CLASSIFICATION
    scoring: ScoringMetric = ScoringMetric.F1

    def info(self) -> str:
        data = [
            ("Penalty (Regularization)", f"{self.penalty}"),
            ("C (Inverse Strength)", f"{self.C}"),
            ("Solver", f"{self.solver}"),
            ("Max Iterations", f"{self.max_iter}"),
            ("Class Weight", f"{self.class_weight}"),
            ("Task Type", f"{self.task_type}"),
            ("Scoring", f"{self.scoring}"),
        ]
        # Include l1_ratio only if relevant to the penalty type
        if self.penalty == "elasticnet":
            data.insert(2, ("L1 Ratio", f"{self.l1_ratio}"))

        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list]:
        """Generate a standard hyperparameter search grid for Logistic Regression.

        The narrow grid targets common defaults, while the expanded grid
        explores broader regularization strengths and solvers.

        Args:
            narrow (bool, optional): If True (default), returns a compact grid.
                If False, returns an expanded grid.

        Returns:
            dict[str, list]: Parameter grid mapping C/penalty/solver (and max_iter
            in expanded mode) to candidate values.

        Example:
            >>> narrow_grid = LogisticRegressionParams.get_standard_search_grid()
            >>> expanded_grid = LogisticRegressionParams.get_standard_search_grid(narrow=False)
        """
        if narrow:
            grid = {"C": [0.1, 1.0, 10.0], "penalty": ["l2"], "solver": ["lbfgs"]}
        else:
            grid = {
                "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "max_iter": [100, 500, 1000],
            }
        return grid


class LogisticRegression(
    ModelSpecification[LogisticRegressionParams, SklearnLogisticRegression]
):
    """Logistic regression model specification."""

    def get_estimator_class(
        self,
    ) -> Type[SklearnLogisticRegression]:
        return SklearnLogisticRegression

    params_class = LogisticRegressionParams

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
    def model_dials(self) -> LogisticRegressionParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: LogisticRegressionParams) -> None:
        self._model_dials = value

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy,
        params: Optional[LogisticRegressionParams] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        scoring: ScoringMetric = ScoringMetric.F1,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        # Use provided params or fall back to defaults
        # We ensure the data's random_state is used if not specified in params
        if params is None:
            params = LogisticRegressionParams(
                task_type=task_type,
                random_state=1,
            )

        self._model_dials = params
        self._task_type = TaskType.CLASSIFICATION
        self._scoring = self.model_dials.scoring

        super().__init__(
            cv=cv,
            balancing_strategy=balancing_strategy,
            n_jobs=n_jobs,
            n_iter=n_iter,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap,
        )

        self._model_type = ModelType.LOGISTIC_REGRESSION
        self.optimization_strategy = optimization_strategy
        self.estimator = self.create_estimator()

    def create_estimator(
        self, parameters: Optional[LogisticRegressionParams] = None
    ) -> SklearnLogisticRegression:
        # Use the provided parameters if they exist, otherwise use the instance dials
        params = parameters or self.model_dials

        return SklearnLogisticRegression(
            penalty=params.penalty,
            C=params.C,
            solver=params.solver,
            max_iter=params.max_iter,
            random_state=params.random_state,
            class_weight=params.class_weight,
            l1_ratio=params.l1_ratio,
        )
