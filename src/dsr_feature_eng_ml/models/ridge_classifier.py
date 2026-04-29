"""Ridge classifier model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from dsr_utils import format_label_value_pairs
from sklearn.linear_model import RidgeClassifier as SklearnRidgeClassifier

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
class RidgeClassifierParams(ClassificationModelParams):
    """
    Hyperparameters for Ridge classifier models.

    Ridge Classifier adapts L2-regularized ridge regression to classification
    by converting class labels to {-1, 1} and treating the task as multi-output
    regression. It is fast, memory-efficient, and competitive on linearly
    separable problems.
    """

    alpha: float | list[float] = 1.0
    class_weight: dict[Any, Any] | Literal["balanced"] | None = None
    solver: Literal[
        "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"
    ] = "auto"
    max_iter: int | None = None
    tol: float = 1e-4
    random_state: int | None = None

    def info(self) -> str:
        """Return a formatted summary of Ridge Classifier parameters."""
        data = [
            ("Alpha", f"{self.alpha}"),
            ("Class Weight", f"{self.class_weight}"),
            ("Solver", f"{self.solver}"),
            ("Max Iter", f"{self.max_iter}"),
            ("Tolerance", f"{self.tol}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Generate a standard hyperparameter search grid for Ridge Classifier.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns a compact grid targeting common alpha values.
            If False, returns a broader search.

        Returns
        -------
        dict[str, list[Any]]
            Parameter grid mapping keys to candidate values.
        """
        if narrow:
            return {"alpha": [0.1, 1.0, 10.0]}

        return {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["auto", "sag", "saga"],
        }


class RidgeClassifierModel(
    ClassificationModelSpecification[RidgeClassifierParams, SklearnRidgeClassifier]
):
    """
    Ridge classifier model specification.

    Manages the lifecycle of a Ridge Classifier, providing standardized
    fitting, tuning, and evaluation through the audit pipeline.
    Ridge Classifier does not support `predict_proba`, so probability
    outputs will be unavailable in classification reports.
    """

    params_class = RidgeClassifierParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: RidgeClassifierParams | None = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        scoring: ScoringMetric = ScoringMetric.ACCURACY,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the Ridge Classifier model specification."""
        if params is None:
            params = RidgeClassifierParams(
                task_type=task_type,
                random_state=1,
                scoring=scoring,
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
        """The Ridge Classifier model type identifier."""
        return ModelType.RIDGE_CLASSIFIER

    @property
    def scoring(self) -> ScoringMetric:
        """The scoring metric used for optimization."""
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_dials(self) -> RidgeClassifierParams:
        """The hyperparameter container for the current model."""
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: RidgeClassifierParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> type[SklearnRidgeClassifier]:
        """Return the scikit-learn RidgeClassifier class."""
        return SklearnRidgeClassifier

    def create_estimator(
        self, parameters: RidgeClassifierParams | None = None
    ) -> SklearnRidgeClassifier:
        """
        Instantiate a raw scikit-learn RidgeClassifier estimator.

        Parameters
        ----------
        parameters : RidgeClassifierParams, optional
            Parameter override. If None, uses instance dials.
        """
        p = parameters or self.model_dials

        alpha = p.alpha[0] if isinstance(p.alpha, list) else p.alpha

        return SklearnRidgeClassifier(
            alpha=alpha,
            class_weight=p.class_weight,
            solver=p.solver,
            max_iter=p.max_iter,
            tol=p.tol,
            random_state=p.random_state,
        )
