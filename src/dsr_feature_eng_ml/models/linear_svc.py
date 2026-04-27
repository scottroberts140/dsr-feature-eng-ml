"""Linear SVC model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

from dsr_utils import format_label_value_pairs
from sklearn.svm import LinearSVC as SklearnLinearSVC

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
class LinearSVCParams(ClassificationModelParams):
    """
    Hyperparameters for Linear Support Vector Classification models.

    Linear SVC fits a linear SVM using a fast primal-based optimization
    (liblinear). It is well-suited to high-dimensional sparse data and
    text classification. Note: LinearSVC does not support ``predict_proba``
    natively; probability calibration would require wrapping with
    ``CalibratedClassifierCV``.
    """

    C: float | list[float] = 1.0
    penalty: Literal["l1", "l2"] = "l2"
    loss: Literal["hinge", "squared_hinge"] = "squared_hinge"
    dual: bool | Literal["auto"] = "auto"
    max_iter: int = 1000
    tol: float = 1e-4
    class_weight: dict[Any, Any] | Literal["balanced"] | None = None
    random_state: int | None = None

    def info(self) -> str:
        """Return a formatted summary of Linear SVC parameters."""
        data = [
            ("C (Regularization)", f"{self.C}"),
            ("Penalty", f"{self.penalty}"),
            ("Loss", f"{self.loss}"),
            ("Max Iter", f"{self.max_iter}"),
            ("Tolerance", f"{self.tol}"),
            ("Class Weight", f"{self.class_weight}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Generate a standard hyperparameter search grid for Linear SVC.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns a compact grid.
            If False, returns an expanded grid.

        Returns
        -------
        dict[str, list[Any]]
            Parameter grid mapping keys to candidate values.
        """
        if narrow:
            return {"C": [0.1, 1.0, 10.0]}

        return {
            "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "loss": ["hinge", "squared_hinge"],
            "max_iter": [1000, 2000, 5000],
        }


class LinearSVCModel(
    ClassificationModelSpecification[LinearSVCParams, SklearnLinearSVC]
):
    """
    Linear SVC model specification.

    Manages the lifecycle of a LinearSVC classifier, providing standardized
    fitting, tuning, and evaluation through the audit pipeline.
    LinearSVC does not support ``predict_proba``, so probability-based
    metrics (ROC-AUC, log-loss) will be unavailable.
    """

    params_class = LinearSVCParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[LinearSVCParams] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        scoring: ScoringMetric = ScoringMetric.ACCURACY,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the Linear SVC model specification."""
        if params is None:
            params = LinearSVCParams(
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
        """The Linear SVC model type identifier."""
        return ModelType.LINEAR_SVC

    @property
    def scoring(self) -> ScoringMetric:
        """The scoring metric used for optimization."""
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_dials(self) -> LinearSVCParams:
        """The hyperparameter container for the current model."""
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: LinearSVCParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> Type[SklearnLinearSVC]:
        """Return the scikit-learn LinearSVC class."""
        return SklearnLinearSVC

    def create_estimator(
        self, parameters: Optional[LinearSVCParams] = None
    ) -> SklearnLinearSVC:
        """
        Instantiate a raw scikit-learn LinearSVC estimator.

        Parameters
        ----------
        parameters : LinearSVCParams, optional
            Parameter override. If None, uses instance dials.
        """
        p = parameters or self.model_dials

        C = p.C[0] if isinstance(p.C, list) else p.C

        return SklearnLinearSVC(
            C=C,
            penalty=p.penalty,
            loss=p.loss,
            dual=p.dual,
            max_iter=p.max_iter,
            tol=p.tol,
            class_weight=p.class_weight,
            random_state=p.random_state,
        )
