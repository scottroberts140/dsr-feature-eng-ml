"""Decision tree model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast, get_args

from dsr_utils import format_label_value_pairs
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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

# Decision Tree Specific Literals
DTCriterion = Literal["gini", "entropy", "log_loss"]
DTRCriterion = Literal["squared_error", "friedman_mse", "absolute_error", "poisson"]

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
class DecisionTreeParams(ModelParams):
    """
    Abstract base hyperparameters for decision tree models.

    Contains all shared fields. Subclasses add task_type, scoring, and
    task-appropriate defaults. Cannot be instantiated directly because
    get_standard_search_grid is not implemented here.
    """

    criterion: str = "gini"
    splitter: Literal["best", "random"] = "best"
    max_depth: int | None | list[int | None] = None
    min_samples_split: int | float | list[int | float] = 2
    min_samples_leaf: int | float | list[int | float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: int | float | Literal["sqrt", "log2"] | None = None
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    class_weight: dict[str, float] | str | None = None
    ccp_alpha: float = 0.0

    def info(self) -> str:
        data = [
            ("Depth", f"{self.max_depth}"),
            ("Alpha (Pruning)", f"{self.ccp_alpha}"),
            ("Min Leaf/Split", f"{self.min_samples_leaf}/{self.min_samples_split}"),
            ("Criterion", f"{self.criterion}"),
            ("Max Features", f"{self.max_features}"),
            ("Weight", f"{self.class_weight}"),
        ]
        return format_label_value_pairs(data)


@dataclass(frozen=True)
class DecisionTreeClassifierParams(DecisionTreeParams):
    """Hyperparameters for classification decision tree models."""

    task_type: TaskType = TaskType.CLASSIFICATION
    scoring: ScoringMetric = ScoringMetric.F1

    def __post_init__(self) -> None:
        """Validate that the criterion is valid for classification."""
        if self.criterion not in get_args(DTCriterion):
            raise ValueError(
                f"Invalid criterion '{self.criterion}' for Classification."
            )

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """Generate standard search space for classification tuning."""
        grid: dict[str, list[Any]] = {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 5, 10],
        }
        if not narrow:
            grid["min_samples_split"] = [2, 5, 10]
            grid["criterion"] = ["gini", "entropy"]
            grid["class_weight"] = [None, "balanced"]
        return grid


@dataclass(frozen=True)
class DecisionTreeRegressorParams(DecisionTreeParams):
    """Hyperparameters for regression decision tree models."""

    criterion: str = "squared_error"
    task_type: TaskType = TaskType.REGRESSION
    scoring: ScoringMetric = ScoringMetric.R2

    def __post_init__(self) -> None:
        """Validate that the criterion is valid for regression."""
        if self.criterion not in get_args(DTRCriterion):
            raise ValueError(f"Invalid criterion '{self.criterion}' for Regression.")
        if self.class_weight is not None:
            raise ValueError("class_weight must be None for Regression tasks.")

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """Generate standard search space for regression tuning."""
        grid: dict[str, list[Any]] = {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 5, 10],
        }
        if not narrow:
            grid["min_samples_split"] = [2, 5, 10]
            grid["criterion"] = ["squared_error", "friedman_mse"]
        return grid


class DecisionTreeClassifierModel(
    ClassificationModelSpecification[
        DecisionTreeClassifierParams, DecisionTreeClassifier
    ]
):
    """Decision tree model specification for classification tasks."""

    params_class = DecisionTreeClassifierParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: DecisionTreeClassifierParams | None = None,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        scoring: ScoringMetric = ScoringMetric.F1,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        if params is None:
            params = DecisionTreeClassifierParams(scoring=scoring, random_state=1)

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
        return ModelType.DECISION_TREE_CLASSIFIER

    @property
    def model_dials(self) -> DecisionTreeClassifierParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: DecisionTreeClassifierParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> type[DecisionTreeClassifier]:
        return DecisionTreeClassifier

    def create_estimator(
        self, parameters: DecisionTreeClassifierParams | None = None
    ) -> DecisionTreeClassifier:
        p = parameters or self.model_dials
        crit = p.criterion if p.criterion in get_args(DTCriterion) else "gini"
        max_depth = _normalize_estimator_param(p.max_depth, "max_depth")
        min_samples_split = _normalize_estimator_param(
            p.min_samples_split, "min_samples_split"
        )
        min_samples_leaf = _normalize_estimator_param(
            p.min_samples_leaf, "min_samples_leaf"
        )
        return DecisionTreeClassifier(
            criterion=cast(DTCriterion, crit),
            class_weight=p.class_weight,
            splitter=p.splitter,
            max_depth=cast(int | None, max_depth),
            min_samples_split=cast(int | float, min_samples_split),
            min_samples_leaf=cast(int | float, min_samples_leaf),
            min_weight_fraction_leaf=p.min_weight_fraction_leaf,
            max_features=p.max_features,
            random_state=p.random_state,
            max_leaf_nodes=p.max_leaf_nodes,
            min_impurity_decrease=p.min_impurity_decrease,
            ccp_alpha=p.ccp_alpha,
        )


class DecisionTreeRegressorModel(
    RegressionModelSpecification[DecisionTreeRegressorParams, DecisionTreeRegressor]
):
    """Decision tree model specification for regression tasks."""

    params_class = DecisionTreeRegressorParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: DecisionTreeRegressorParams | None = None,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        scoring: ScoringMetric = ScoringMetric.R2,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        if params is None:
            params = DecisionTreeRegressorParams(scoring=scoring, random_state=1)

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
        return ModelType.DECISION_TREE_REGRESSOR

    @property
    def model_dials(self) -> DecisionTreeRegressorParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: DecisionTreeRegressorParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> type[DecisionTreeRegressor]:
        return DecisionTreeRegressor

    def create_estimator(
        self, parameters: DecisionTreeRegressorParams | None = None
    ) -> DecisionTreeRegressor:
        p = parameters or self.model_dials
        crit = p.criterion if p.criterion in get_args(DTRCriterion) else "squared_error"
        max_depth = _normalize_estimator_param(p.max_depth, "max_depth")
        min_samples_split = _normalize_estimator_param(
            p.min_samples_split, "min_samples_split"
        )
        min_samples_leaf = _normalize_estimator_param(
            p.min_samples_leaf, "min_samples_leaf"
        )
        return DecisionTreeRegressor(
            criterion=cast(DTRCriterion, crit),
            splitter=p.splitter,
            max_depth=cast(int | None, max_depth),
            min_samples_split=cast(int | float, min_samples_split),
            min_samples_leaf=cast(int | float, min_samples_leaf),
            min_weight_fraction_leaf=p.min_weight_fraction_leaf,
            max_features=p.max_features,
            random_state=p.random_state,
            max_leaf_nodes=p.max_leaf_nodes,
            min_impurity_decrease=p.min_impurity_decrease,
            ccp_alpha=p.ccp_alpha,
        )
