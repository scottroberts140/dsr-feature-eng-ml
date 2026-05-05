"""Random forest model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeVar, cast, get_args

from dsr_utils import format_label_value_pairs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.models.model_specification import (
    ClassificationModelSpecification,
    ModelParams,
    RegressionModelSpecification,
)
from dsr_feature_eng_ml.prefs_instance import prefs

# Literal types for scikit-learn constructor validation
RFCriterion = Literal["gini", "entropy", "log_loss"]
RFRCriterion = Literal["squared_error", "absolute_error", "friedman_mse", "poisson"]

T = TypeVar("T")


def _normalize_estimator_param(value: T | list[T] | tuple[T, ...], name: str) -> T:
    """Return a scalar value for estimator construction.

    When model params are configured as search spaces (lists/tuples),
    scikit-learn estimators still require scalar constructor args.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"Parameter '{name}' cannot be an empty list or tuple.")
        return value[0]
    return value


@dataclass(frozen=True)
class RandomForestParams(ModelParams):
    """
    Abstract base hyperparameters for Random Forest models.

    Contains all shared fields. Subclasses add task_type, scoring, and
    task-appropriate defaults. Cannot be instantiated directly because
    get_standard_search_grid is not implemented here.
    """

    criterion: str | list[str] = "gini"
    max_depth: int | None | list[int | None] = None
    n_estimators: int | list[int] = 100
    min_samples_split: int | float | list[int | float] = 2
    min_samples_leaf: int | float | list[int | float] = 1
    max_features: float | Literal["sqrt", "log2"] | None = "sqrt"
    bootstrap: bool = True

    # For classification models only
    class_weight: (
        Mapping[Any, Any] | Literal["balanced", "balanced_subsample"] | None
    ) = None

    def info(self) -> str:
        """Return a formatted summary of Random Forest parameters."""
        data = [
            ("Depth", f"{self.max_depth}"),
            ("Estimators", f"{self.n_estimators}"),
            ("Min Leaf/Split", f"{self.min_samples_leaf}/{self.min_samples_split}"),
            ("Criterion", f"{self.criterion}"),
            ("Max Features", f"{self.max_features}"),
            ("Bootstrap", f"{self.bootstrap}"),
            ("Weight", f"{self.class_weight}"),
        ]
        return format_label_value_pairs(data)


@dataclass(frozen=True)
class RandomForestClassifierParams(RandomForestParams):
    """Hyperparameters for classification random forest models."""

    task_type: TaskType = TaskType.CLASSIFICATION
    scoring: ScoringMetric = ScoringMetric.F1

    def __post_init__(self) -> None:
        """Validate that criterion values are valid for classification."""
        valid = set(get_args(RFCriterion))
        criteria = (
            list(self.criterion)
            if isinstance(self.criterion, (list, tuple))
            else [self.criterion]
        )
        invalid = [c for c in criteria if c not in valid]
        if invalid:
            raise ValueError(
                f"Invalid criterion value(s) {invalid} for Classification. "
                f"Expected values from {sorted(valid)}"
            )

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """Generate a standard search grid for classification tuning."""
        if narrow:
            return {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "max_features": ["sqrt", "log2"],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True],
            }
        return {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [None, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 10],
            "max_features": ["sqrt", "log2", None, 0.5],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
        }


@dataclass(frozen=True)
class RandomForestRegressorParams(RandomForestParams):
    """Hyperparameters for regression random forest models."""

    criterion: str | list[str] = "squared_error"
    max_features: float | Literal["sqrt", "log2"] | None = 1.0
    task_type: TaskType = TaskType.REGRESSION
    scoring: ScoringMetric = ScoringMetric.R2

    def __post_init__(self) -> None:
        """Validate that criterion values are valid for regression."""
        valid = set(get_args(RFRCriterion))
        criteria = (
            list(self.criterion)
            if isinstance(self.criterion, (list, tuple))
            else [self.criterion]
        )
        invalid = [c for c in criteria if c not in valid]
        if invalid:
            raise ValueError(
                f"Invalid criterion value(s) {invalid} for Regression. "
                f"Expected values from {sorted(valid)}"
            )

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """Generate a standard search grid for regression tuning."""
        if narrow:
            return {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "max_features": ["sqrt", "log2"],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True],
            }
        return {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [None, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 10],
            "max_features": ["sqrt", "log2", None, 0.5],
            "bootstrap": [True, False],
            "criterion": ["squared_error", "friedman_mse"],
        }


class RandomForestClassifierModel(
    ClassificationModelSpecification[
        RandomForestClassifierParams, RandomForestClassifier
    ]
):
    """Random forest model specification for classification tasks."""

    params_class = RandomForestClassifierParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: RandomForestClassifierParams | None = None,
        scoring: ScoringMetric = ScoringMetric.F1,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        if params is None:
            params = RandomForestClassifierParams(scoring=scoring, random_state=1)

        self._model_dials = params
        self._scoring = scoring

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
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_type(self) -> ModelType:
        return ModelType.RANDOM_FOREST_CLASSIFIER

    @property
    def model_dials(self) -> RandomForestClassifierParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: RandomForestClassifierParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> type[RandomForestClassifier]:
        """Return the Scikit-Learn RandomForestClassifier class."""
        return RandomForestClassifier

    def create_estimator(
        self, parameters: RandomForestClassifierParams | None = None
    ) -> RandomForestClassifier:
        """Instantiate a raw Scikit-Learn RandomForestClassifier estimator."""
        p = parameters or self.model_dials
        crit = p.criterion if p.criterion in get_args(RFCriterion) else "gini"
        n_estimators = _normalize_estimator_param(p.n_estimators, "n_estimators")
        max_depth = _normalize_estimator_param(p.max_depth, "max_depth")
        min_samples_split = _normalize_estimator_param(
            p.min_samples_split, "min_samples_split"
        )
        min_samples_leaf = _normalize_estimator_param(
            p.min_samples_leaf, "min_samples_leaf"
        )
        return RandomForestClassifier(
            criterion=cast(RFCriterion, crit),
            class_weight=p.class_weight,
            n_estimators=cast(int, n_estimators),
            max_depth=cast(int | None, max_depth),
            min_samples_split=cast(int | float, min_samples_split),
            min_samples_leaf=cast(int | float, min_samples_leaf),
            max_features=cast(float | Literal["sqrt", "log2"], p.max_features),
            random_state=p.random_state,
            bootstrap=p.bootstrap,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )


class RandomForestRegressorModel(
    RegressionModelSpecification[RandomForestRegressorParams, RandomForestRegressor]
):
    """Random forest model specification for regression tasks."""

    params_class = RandomForestRegressorParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: RandomForestRegressorParams | None = None,
        scoring: ScoringMetric = ScoringMetric.R2,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        if params is None:
            params = RandomForestRegressorParams(scoring=scoring, random_state=1)

        self._model_dials = params
        self._scoring = scoring

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
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_type(self) -> ModelType:
        return ModelType.RANDOM_FOREST_REGRESSOR

    @property
    def model_dials(self) -> RandomForestRegressorParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: RandomForestRegressorParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> type[RandomForestRegressor]:
        """Return the Scikit-Learn RandomForestRegressor class."""
        return RandomForestRegressor

    def create_estimator(
        self, parameters: RandomForestParams | None = None
    ) -> RandomForestRegressor:
        """Instantiate a raw Scikit-Learn RandomForestRegressor estimator."""
        p = parameters or self.model_dials
        crit = p.criterion if p.criterion in get_args(RFRCriterion) else "squared_error"
        n_estimators = _normalize_estimator_param(p.n_estimators, "n_estimators")
        max_depth = _normalize_estimator_param(p.max_depth, "max_depth")
        min_samples_split = _normalize_estimator_param(
            p.min_samples_split, "min_samples_split"
        )
        min_samples_leaf = _normalize_estimator_param(
            p.min_samples_leaf, "min_samples_leaf"
        )
        return RandomForestRegressor(
            criterion=cast(RFRCriterion, crit),
            n_estimators=cast(int, n_estimators),
            max_depth=cast(int | None, max_depth),
            min_samples_split=cast(int | float, min_samples_split),
            min_samples_leaf=cast(int | float, min_samples_leaf),
            max_features=cast(float | Literal["sqrt", "log2"], p.max_features),
            random_state=p.random_state,
            bootstrap=p.bootstrap,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
