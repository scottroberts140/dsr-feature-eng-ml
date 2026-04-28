"""K-Nearest Neighbors classifier model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

from dsr_utils import format_label_value_pairs
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier

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
class KNeighborsClassifierParams(ClassificationModelParams):
    """
    Hyperparameters for K-Nearest Neighbors classifier models.

    KNN classifies samples by a majority vote among their k nearest neighbors
    in feature space. It is non-parametric and instance-based, requiring no
    training phase, but can be slow at prediction time for large datasets.
    Feature scaling is important for distance-based methods.
    """

    n_neighbors: int | list[int] = 5
    weights: Literal["uniform", "distance"] | list[Literal["uniform", "distance"]] = (
        "uniform"
    )
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: int = 2  # 1 = Manhattan, 2 = Euclidean
    metric: str = "minkowski"

    def info(self) -> str:
        """Return a formatted summary of KNN parameters."""
        data = [
            ("N Neighbors", f"{self.n_neighbors}"),
            ("Weights", f"{self.weights}"),
            ("Algorithm", f"{self.algorithm}"),
            ("Leaf Size", f"{self.leaf_size}"),
            ("Distance Metric (p)", f"{self.p}"),
            ("Metric", f"{self.metric}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Generate a standard hyperparameter search grid for KNN.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns a compact grid.
            If False, returns a broader search over k and distance metrics.

        Returns
        -------
        dict[str, list[Any]]
            Parameter grid mapping keys to candidate values.
        """
        if narrow:
            return {"n_neighbors": [3, 5, 11], "weights": ["uniform", "distance"]}

        return {
            "n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31, 51],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
        }


class KNeighborsClassifierModel(
    ClassificationModelSpecification[
        KNeighborsClassifierParams, SklearnKNeighborsClassifier
    ]
):
    """
    K-Nearest Neighbors classifier model specification.

    Manages the lifecycle of a KNeighborsClassifier, providing standardized
    fitting, tuning, and evaluation through the audit pipeline.
    KNN supports ``predict_proba`` so probability-based metrics are available.
    Note that KNN can be memory- and time-intensive on large datasets; the
    tuning multiplier is set high (25.0) to reflect this.
    """

    params_class = KNeighborsClassifierParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[KNeighborsClassifierParams] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        scoring: ScoringMetric = ScoringMetric.F1,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the KNN classifier model specification."""
        if params is None:
            params = KNeighborsClassifierParams(
                task_type=task_type,
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
        """The K-Nearest Neighbors classifier model type identifier."""
        return ModelType.K_NEIGHBORS_CLASSIFIER

    @property
    def scoring(self) -> ScoringMetric:
        """The scoring metric used for optimization."""
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_dials(self) -> KNeighborsClassifierParams:
        """The hyperparameter container for the current model."""
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: KNeighborsClassifierParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> Type[SklearnKNeighborsClassifier]:
        """Return the scikit-learn KNeighborsClassifier class."""
        return SklearnKNeighborsClassifier

    def create_estimator(
        self, parameters: Optional[KNeighborsClassifierParams] = None
    ) -> SklearnKNeighborsClassifier:
        """
        Instantiate a raw scikit-learn KNeighborsClassifier estimator.

        Parameters
        ----------
        parameters : KNeighborsClassifierParams, optional
            Parameter override. If None, uses instance dials.
        """
        p = parameters or self.model_dials

        n_neighbors = (
            p.n_neighbors[0] if isinstance(p.n_neighbors, list) else p.n_neighbors
        )
        weights = p.weights[0] if isinstance(p.weights, list) else p.weights

        return SklearnKNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=p.algorithm,
            leaf_size=p.leaf_size,
            p=p.p,
            metric=p.metric,
            n_jobs=self.n_jobs,
        )
