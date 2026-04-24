"""Logistic regression model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

from dsr_utils import format_label_value_pairs
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.models.model_specification import (
    ClassificationModelParams,
    ClassificationModelSpecification,
)
from dsr_feature_eng_ml.prefs_instance import prefs


@dataclass(frozen=True)
class LogisticRegressionParams(ClassificationModelParams):
    """
    Hyperparameters for logistic regression models.

    Logistic Regression is a linear model for classification. It estimates
    probabilities using a logistic function to handle binary or multiclass
    dependent variables, supporting various regularization penalties.
    """

    penalty: Literal["l1", "l2", "elasticnet", None] = "l2"
    C: float = 1.0
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs"
    max_iter: int = 100
    class_weight: dict[Any, Any] | str | None = None
    l1_ratio: float | None = None  # Only used if penalty='elasticnet'
    random_state: int | None = None

    def info(self) -> str:
        """Return a formatted summary of Logistic Regression parameters."""
        data = [
            ("Penalty (Regularization)", f"{self.penalty}"),
            ("C (Inverse Strength)", f"{self.C}"),
            ("Solver", f"{self.solver}"),
            ("Max Iterations", f"{self.max_iter}"),
            ("Class Weight", f"{self.class_weight}"),
        ]
        # Include l1_ratio only if relevant to the penalty type
        if self.penalty == "elasticnet":
            data.insert(2, ("L1 Ratio", f"{self.l1_ratio}"))

        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Generate a standard hyperparameter search grid for Logistic Regression.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns a compact grid targeting common defaults.
            If False, returns an expanded grid exploring broader strengths
            and solvers.

        Returns
        -------
        dict[str, list[Any]]
            A parameter grid mapping keys to candidate values.
        """
        if narrow:
            return {"C": [0.1, 1.0, 10.0], "l1_ratio": [0.0], "solver": ["lbfgs"]}

        return {
            "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "l1_ratio": [0.0, 1.0],
            "solver": ["liblinear", "saga"],
            "max_iter": [100, 500, 1000],
        }


class LogisticRegression(
    ClassificationModelSpecification[
        LogisticRegressionParams, SklearnLogisticRegression
    ]
):
    """
    Logistic regression model specification.

    This class manages the lifecycle of a Logistic Regression classifier,
    providing standardized fitting, tuning, and evaluation.
    """

    params_class = LogisticRegressionParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[LogisticRegressionParams] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        scoring: ScoringMetric = ScoringMetric.F1,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the Logistic Regression model specification."""
        if params is None:
            params = LogisticRegressionParams(
                task_type=task_type, random_state=1, scoring=scoring
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
            optimization_strategy=optimization_strategy,
        )

        self.estimator = self.create_estimator()

    @property
    def model_type(self) -> ModelType:
        """The Logistic Regression model type identifier."""
        return ModelType.LOGISTIC_REGRESSION

    @property
    def scoring(self) -> ScoringMetric:
        """The scoring metric used for optimization."""
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_dials(self) -> LogisticRegressionParams:
        """The hyperparameter container for the current model."""
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: LogisticRegressionParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> Type[SklearnLogisticRegression]:
        """Return the Scikit-Learn LogisticRegression class."""
        return SklearnLogisticRegression

    def create_estimator(
        self, parameters: Optional[LogisticRegressionParams] = None
    ) -> SklearnLogisticRegression:
        """
        Instantiate a raw Scikit-Learn LogisticRegression estimator.

        The ``penalty`` field on ``LogisticRegressionParams`` is translated to
        the ``l1_ratio`` parameter required by scikit-learn 1.8+:
        ``"l2"`` → 0.0, ``"l1"`` → 1.0, ``"elasticnet"`` uses the explicit
        ``l1_ratio`` value, and ``None`` disables regularization.

        Parameters
        ----------
        parameters : LogisticRegressionParams, optional
            Parameter override. If None, uses instance dials.
        """
        p = parameters or self.model_dials

        # sklearn 1.8+: penalty param deprecated; use l1_ratio instead.
        # l1_ratio=0.0 → L2, l1_ratio=1.0 → L1, l1_ratio in (0,1) → elasticnet, None → no regularization.
        _penalty_to_l1_ratio: dict[str, float] = {"l2": 0.0, "l1": 1.0}
        if p.penalty is None:
            l1_ratio = None
        elif p.penalty == "elasticnet":
            l1_ratio = p.l1_ratio
        else:
            l1_ratio = _penalty_to_l1_ratio.get(p.penalty, 0.0)

        return SklearnLogisticRegression(
            l1_ratio=l1_ratio,
            C=p.C,
            solver=p.solver,
            max_iter=p.max_iter,
            random_state=p.random_state,
            class_weight=p.class_weight,
        )
