"""Random forest model specification and parameter definitions."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Type, cast, get_args

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
    ModelParams,
    ModelSpecification,
)
from dsr_feature_eng_ml.prefs_instance import prefs

# Literal types for scikit-learn constructor validation
RFCriterion = Literal["gini", "entropy", "log_loss"]
RFRCriterion = Literal["squared_error", "absolute_error", "friedman_mse", "poisson"]


@dataclass(frozen=True)
class RandomForestParams(ModelParams):
    """
    Hyperparameters for Random Forest models.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    """

    criterion: str = "squared_error"
    max_depth: int | None = None
    n_estimators: int = 100
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    max_features: float | Literal["sqrt", "log2"] | None = "sqrt"
    bootstrap: bool = True
    task_type: TaskType = TaskType.CLASSIFICATION
    scoring: ScoringMetric = ScoringMetric.F1

    # For classification models only
    class_weight: (
        Mapping[Any, Any] | Literal["balanced", "balanced_subsample"] | None
    ) = None

    @classmethod
    def create_default(
        cls,
        task_type: TaskType,
        scoring: ScoringMetric,
        random_state: Optional[int],
        **kwargs: Any,
    ) -> RandomForestParams:
        """Create a parameter instance with task-appropriate defaults."""
        if task_type == TaskType.REGRESSION:
            defaults = {
                "task_type": task_type,
                "criterion": "squared_error",
                "max_features": 1.0,  # Typically better for regression
                "random_state": random_state,
                "scoring": scoring,
                **kwargs,
            }
        else:
            defaults = {
                "task_type": task_type,
                "criterion": "gini",
                "max_features": "sqrt",
                "random_state": random_state,
                "scoring": scoring,
                **kwargs,
            }
        return cls(**defaults)

    def __post_init__(self) -> None:
        """Validate that the criterion matches the task type."""
        if self.task_type == TaskType.CLASSIFICATION:
            if self.criterion not in get_args(RFCriterion):
                raise ValueError(
                    f"Invalid criterion '{self.criterion}' for Classification. "
                    f"Expected one of {get_args(RFCriterion)}"
                )
        else:
            if self.criterion not in get_args(RFRCriterion):
                raise ValueError(
                    f"Invalid criterion '{self.criterion}' for Regression. "
                    f"Expected one of {get_args(RFRCriterion)}"
                )

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

    @staticmethod
    def get_standard_search_grid(
        narrow: bool = True, task_type: TaskType = TaskType.CLASSIFICATION
    ) -> dict[str, list[Any]]:
        """
        Generate a standard search grid for Random Forest tuning.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns core parameters. If False, adds split thresholds,
            bootstrap toggles, and criteria.
        task_type : TaskType, default TaskType.CLASSIFICATION
            Determines which criteria are included in the grid.
        """
        if narrow:
            return {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "max_features": ["sqrt", "log2"],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True],
            }

        grid = {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [None, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4, 10],
            "max_features": ["sqrt", "log2", None, 0.5],
            "bootstrap": [True, False],
        }

        if task_type == TaskType.CLASSIFICATION:
            grid["criterion"] = ["gini", "entropy"]
        else:
            grid["criterion"] = ["squared_error", "friedman_mse"]

        return grid


class RandomForest(
    ModelSpecification[
        RandomForestParams, RandomForestClassifier | RandomForestRegressor
    ]
):
    """Random forest specification for regression and classification."""

    params_class = RandomForestParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[RandomForestParams] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        scoring: Optional[ScoringMetric] = None,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the Random Forest model specification."""
        resolved_scoring = scoring or (
            ScoringMetric.R2 if task_type == TaskType.REGRESSION else ScoringMetric.F1
        )

        if params is None:
            params = RandomForestParams.create_default(
                task_type=task_type, scoring=resolved_scoring, random_state=1
            )
        else:
            if params.task_type != task_type or (
                scoring is not None and params.scoring != resolved_scoring
            ):
                params = dataclasses.replace(
                    params,
                    task_type=task_type,
                    scoring=resolved_scoring,
                )

        self._model_dials = params
        self._task_type = self.model_dials.task_type
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

        self._model_type = (
            ModelType.RANDOM_FOREST_CLASSIFIER
            if self.task_type == TaskType.CLASSIFICATION
            else ModelType.RANDOM_FOREST_REGRESSOR
        )
        self.estimator = self.create_estimator()

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def model_type(self) -> ModelType:
        return self._model_type

    @property
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_dials(self) -> RandomForestParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: RandomForestParams) -> None:
        self._model_dials = value

    def get_estimator_class(
        self,
    ) -> Type[RandomForestClassifier | RandomForestRegressor]:
        """Return the Scikit-Learn Random Forest class for the current task."""
        return (
            RandomForestClassifier
            if self.task_type == TaskType.CLASSIFICATION
            else RandomForestRegressor
        )

    def create_estimator(
        self, parameters: Optional[RandomForestParams] = None
    ) -> RandomForestClassifier | RandomForestRegressor:
        """Instantiate a raw Scikit-Learn Random Forest estimator."""
        p = parameters or self.model_dials

        common = {
            "n_estimators": p.n_estimators,
            "max_depth": p.max_depth,
            "min_samples_split": p.min_samples_split,
            "min_samples_leaf": p.min_samples_leaf,
            "max_features": p.max_features,
            "random_state": p.random_state,
            "bootstrap": p.bootstrap,
            "n_jobs": self.n_jobs,
        }

        if self.task_type == TaskType.REGRESSION:
            crit = (
                p.criterion
                if p.criterion in get_args(RFRCriterion)
                else "squared_error"
            )
            return RandomForestRegressor(criterion=cast(RFRCriterion, crit), **common)

        crit = p.criterion if p.criterion in get_args(RFCriterion) else "gini"
        return RandomForestClassifier(
            criterion=cast(RFCriterion, crit), class_weight=p.class_weight, **common
        )


class RandomForestClassifierModel(RandomForest):
    """Task-specific random forest wrapper for classification models."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[RandomForestParams] = None,
        scoring: ScoringMetric = ScoringMetric.F1,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        if params is not None and params.task_type != TaskType.CLASSIFICATION:
            raise ValueError(
                "RandomForestClassifierModel requires params.task_type == TaskType.CLASSIFICATION"
            )

        super().__init__(
            cv=cv,
            balancing_strategy=balancing_strategy,
            params=params,
            task_type=TaskType.CLASSIFICATION,
            scoring=scoring,
            n_jobs=n_jobs,
            n_iter=n_iter,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap,
            optimization_strategy=optimization_strategy,
        )


class RandomForestRegressorModel(RandomForest):
    """Task-specific random forest wrapper for regression models."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[RandomForestParams] = None,
        scoring: ScoringMetric = ScoringMetric.R2,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        if params is not None and params.task_type != TaskType.REGRESSION:
            raise ValueError(
                "RandomForestRegressorModel requires params.task_type == TaskType.REGRESSION"
            )

        super().__init__(
            cv=cv,
            balancing_strategy=balancing_strategy,
            params=params,
            task_type=TaskType.REGRESSION,
            scoring=scoring,
            n_jobs=n_jobs,
            n_iter=n_iter,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap,
            optimization_strategy=optimization_strategy,
        )
