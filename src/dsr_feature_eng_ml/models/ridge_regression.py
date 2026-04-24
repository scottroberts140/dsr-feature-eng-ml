"""Ridge regression model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type, cast

from dsr_utils import format_label_value_pairs
from sklearn.linear_model import Ridge as SklearnRidge

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
class RidgeParams(RegressionModelParams):
    """
    Hyperparameters for ridge regression models.

    Ridge regression addresses some of the problems of Ordinary Least Squares
    by imposing a penalty on the size of the coefficients with L2 regularization.
    This results in a model that is more robust to collinearity.
    """

    alpha: float = 1.0
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: int | None = None
    tol: float = 1e-4
    solver: Literal[
        "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"
    ] = "auto"
    random_state: int | None = None

    def info(self) -> str:
        """Return a formatted summary of Ridge parameters."""
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
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Generate a standard hyperparameter search grid for Ridge Regression.

        Regularization strength is explored logarithmically to cover different
        orders of magnitude efficiently.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns a compact grid (alpha: 0.1 to 100).
            If False, returns a wider range (alpha: 0.001 to 1000).

        Returns
        -------
        dict[str, list[Any]]
            A parameter grid mapping 'alpha' to candidate values.
        """
        if narrow:
            return {"alpha": [0.1, 1.0, 10.0, 100.0]}

        return {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}


class RidgeRegression(RegressionModelSpecification[RidgeParams, SklearnRidge]):
    """
    Ridge regression model specification.

    This class handles the lifecycle of a Ridge model, providing standardized
    fitting, tuning, and evaluation for regression tasks.
    """

    params_class = RidgeParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[RidgeParams] = None,
        task_type: TaskType = TaskType.REGRESSION,
        scoring: ScoringMetric = ScoringMetric.R2,
        n_jobs: int = -1,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the Ridge regression model specification."""
        if params is None:
            params = RidgeParams(task_type=task_type, random_state=1, scoring=scoring)

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
        """The Ridge model type identifier."""
        return ModelType.RIDGE

    @property
    def scoring(self) -> ScoringMetric:
        """The scoring metric used for optimization."""
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_dials(self) -> RidgeParams:
        """The hyperparameter container for the current model."""
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: RidgeParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> Type[SklearnRidge]:
        """Return the Scikit-Learn Ridge class."""
        return SklearnRidge

    def create_estimator(
        self, parameters: Optional[RidgeParams] = None
    ) -> SklearnRidge:
        """
        Instantiate a raw Scikit-Learn Ridge estimator.

        Parameters
        ----------
        parameters : RidgeParams, optional
            Parameter override. If None, uses instance dials.
        """
        p = parameters or self.model_dials

        return SklearnRidge(
            alpha=p.alpha,
            fit_intercept=p.fit_intercept,
            copy_X=p.copy_X,
            max_iter=p.max_iter,
            tol=p.tol,
            solver=cast(
                Literal[
                    "auto",
                    "svd",
                    "cholesky",
                    "lsqr",
                    "sparse_cg",
                    "sag",
                    "saga",
                    "lbfgs",
                ],
                p.solver,
            ),
            random_state=p.random_state,
        )
