"""Decision tree model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Type, cast, get_args

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


@dataclass(frozen=True)
class DecisionTreeParams(ModelParams):
    """
    Hyperparameters for decision tree models.
    """

    criterion: str = "gini"
    splitter: Literal["best", "random"] = "best"
    max_depth: Optional[int] = None
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: int | float | Literal["sqrt", "log2"] | None = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    class_weight: dict[str, float] | str | None = None
    ccp_alpha: float = 0.0
    task_type: TaskType = TaskType.CLASSIFICATION
    scoring: ScoringMetric = ScoringMetric.F1

    @classmethod
    def create_default(
        cls,
        task_type: TaskType,
        scoring: ScoringMetric,
        random_state: Optional[int],
        **kwargs: Any,
    ) -> DecisionTreeParams:
        """Create a parameter instance with task-appropriate defaults."""
        if task_type == TaskType.REGRESSION:
            defaults = {
                "task_type": task_type,
                "criterion": "squared_error",
                "random_state": random_state,
                "scoring": scoring,
                **kwargs,
            }
        else:
            defaults = {
                "task_type": task_type,
                "criterion": "gini",
                "random_state": random_state,
                "scoring": scoring,
                **kwargs,
            }
        return cls(**defaults)

    def __post_init__(self) -> None:
        """Validate that the criterion matches the task type."""
        if self.task_type == TaskType.CLASSIFICATION:
            if self.criterion not in get_args(DTCriterion):
                raise ValueError(
                    f"Invalid criterion '{self.criterion}' for Classification."
                )
        else:
            if self.criterion not in get_args(DTRCriterion):
                raise ValueError(
                    f"Invalid criterion '{self.criterion}' for Regression."
                )
            if self.class_weight is not None:
                raise ValueError("class_weight must be None for Regression tasks.")

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

    @staticmethod
    def get_standard_search_grid(
        narrow: bool = True, task_type: TaskType = TaskType.CLASSIFICATION
    ) -> dict[str, list[Any]]:
        """Generate standard search space for tuning."""
        grid: dict[str, list[Any]] = {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 5, 10],
        }

        if not narrow:
            grid["min_samples_split"] = [2, 5, 10]
            if task_type == TaskType.CLASSIFICATION:
                grid["criterion"] = ["gini", "entropy"]
                grid["class_weight"] = [None, "balanced"]
            else:
                grid["criterion"] = ["squared_error", "friedman_mse"]

        return grid


class DecisionTreeClassifierModel(
    ClassificationModelSpecification[DecisionTreeParams, DecisionTreeClassifier]
):
    """Decision tree model specification for classification tasks."""

    params_class = DecisionTreeParams

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[DecisionTreeParams] = None,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        scoring: ScoringMetric = ScoringMetric.F1,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        if params is not None and params.task_type != TaskType.CLASSIFICATION:
            raise ValueError(
                "DecisionTreeClassifierModel requires params.task_type == TaskType.CLASSIFICATION"
            )

        if params is None:
            params = DecisionTreeParams.create_default(
                task_type=TaskType.CLASSIFICATION, scoring=scoring, random_state=1
            )

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

        self._model_type = ModelType.DECISION_TREE_CLASSIFIER
        self.estimator = self.create_estimator()

    @property
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def model_dials(self) -> DecisionTreeParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: DecisionTreeParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> Type[DecisionTreeClassifier]:
        return DecisionTreeClassifier

    def create_estimator(
        self, parameters: Optional[DecisionTreeParams] = None
    ) -> DecisionTreeClassifier:
        p = parameters or self.model_dials
        crit = p.criterion if p.criterion in get_args(DTCriterion) else "gini"
        return DecisionTreeClassifier(
            criterion=cast(DTCriterion, crit),
            class_weight=p.class_weight,
            splitter=p.splitter,
            max_depth=p.max_depth,
            min_samples_split=p.min_samples_split,
            min_samples_leaf=p.min_samples_leaf,
            min_weight_fraction_leaf=p.min_weight_fraction_leaf,
            max_features=p.max_features,
            random_state=p.random_state,
            max_leaf_nodes=p.max_leaf_nodes,
            min_impurity_decrease=p.min_impurity_decrease,
            ccp_alpha=p.ccp_alpha,
        )


class DecisionTreeRegressorModel(
    RegressionModelSpecification[DecisionTreeParams, DecisionTreeRegressor]
):
    """Decision tree model specification for regression tasks."""

    params_class = DecisionTreeParams

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[DecisionTreeParams] = None,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        scoring: ScoringMetric = ScoringMetric.R2,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        if params is not None and params.task_type != TaskType.REGRESSION:
            raise ValueError(
                "DecisionTreeRegressorModel requires params.task_type == TaskType.REGRESSION"
            )

        if params is None:
            params = DecisionTreeParams.create_default(
                task_type=TaskType.REGRESSION, scoring=scoring, random_state=1
            )

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

        self._model_type = ModelType.DECISION_TREE_REGRESSOR
        self.estimator = self.create_estimator()

    @property
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def model_dials(self) -> DecisionTreeParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: DecisionTreeParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> Type[DecisionTreeRegressor]:
        return DecisionTreeRegressor

    def create_estimator(
        self, parameters: Optional[DecisionTreeParams] = None
    ) -> DecisionTreeRegressor:
        p = parameters or self.model_dials
        crit = (
            p.criterion if p.criterion in get_args(DTRCriterion) else "squared_error"
        )
        return DecisionTreeRegressor(
            criterion=cast(DTRCriterion, crit),
            splitter=p.splitter,
            max_depth=p.max_depth,
            min_samples_split=p.min_samples_split,
            min_samples_leaf=p.min_samples_leaf,
            min_weight_fraction_leaf=p.min_weight_fraction_leaf,
            max_features=p.max_features,
            random_state=p.random_state,
            max_leaf_nodes=p.max_leaf_nodes,
            min_impurity_decrease=p.min_impurity_decrease,
            ccp_alpha=p.ccp_alpha,
        )
