from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional, Literal, Mapping, Any, cast, get_args, Type
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RFCriterion = Literal["gini", "entropy", "log_loss"]
RFRCriterion = Literal["squared_error", "absolute_error", "friedman_mse", "poisson"]


@dataclass(frozen=True)
class RandomForestParams(ModelParams):
    criterion: Union[RFCriterion, RFRCriterion] = "squared_error"
    max_depth: Optional[int] = None
    n_estimators: int = 100
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    max_features: Union[float, Literal["sqrt", "log2"]] = "sqrt"
    random_state: Optional[int] = None
    bootstrap: bool = True
    task_type: TaskType = TaskType.CLASSIFICATION
    scoring: ScoringMetric = ScoringMetric.R2

    # For classification models only
    class_weight: Optional[
        Union[Mapping[Any, Any], Literal["balanced", "balanced_subsample"]]
    ] = None

    @classmethod
    def create_default(
        cls, task_type: TaskType, scoring: ScoringMetric, random_state: int, **kwargs
    ) -> RandomForestParams:
        if task_type == TaskType.REGRESSION:
            defaults = {
                "task_type": task_type,
                "criterion": "squared_error",
                "max_features": 1.0,  # Often better for regression
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

    def __post_init__(self):
        """Validates that the criterion matches the task type."""
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
        data = [
            ("Depth", f"{self.max_depth}"),
            ("Estimators", f"{self.n_estimators}"),
            ("Min Leaf/Split", f"{self.min_samples_leaf}/{self.min_samples_split}"),
            ("Criterion", f"{self.criterion}"),
            ("Max Features", f"{self.max_features}"),
            ("Bootstrap", f"{self.bootstrap}"),
            ("Weight", f"{self.class_weight}"),
            ("Task Type", f"{self.task_type}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list]:
        if narrow:
            grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "max_features": ["sqrt", "log2"],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True],
            }
        else:
            grid = {
                "n_estimators": [100, 200, 500, 1000],
                "max_depth": [None, 10, 20, 30, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4, 10],
                "max_features": ["sqrt", "log2", None, 0.5],
                "criterion": ["gini", "entropy"],
                "bootstrap": [True, False],
                "max_samples": [0.5, 0.7, 0.9, None],  # Only used if bootstrap=True
            }

        return grid


class RandomForest(
    ModelSpecification[
        RandomForestParams, Union[RandomForestClassifier, RandomForestRegressor]
    ]
):
    def get_estimator_class(
        self,
    ) -> Type[Union[RandomForestClassifier, RandomForestRegressor]]:
        if self.task_type == TaskType.CLASSIFICATION:
            return RandomForestClassifier
        return RandomForestRegressor

    params_class = RandomForestParams

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
    def scoring(self, value: ScoringMetric):
        self._scoring = value

    @property
    def model_dials(self) -> RandomForestParams:
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: RandomForestParams) -> None:
        self._model_dials = value

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy,
        params: Optional[RandomForestParams] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        scoring: ScoringMetric = ScoringMetric.R2,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        # Use provided params or fall back to defaults
        # We ensure the data's random_state is used if not specified in params
        if params is None:
            params = RandomForestParams.create_default(
                task_type=task_type,
                scoring=scoring,
                random_state=1,
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
        )

        if task_type == TaskType.CLASSIFICATION:
            self._model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        else:
            self._model_type = ModelType.RANDOM_FOREST_REGRESSOR

        self.optimization_strategy = optimization_strategy
        self.estimator = self.create_estimator()

    def create_estimator(
        self, parameters: Optional[RandomForestParams] = None
    ) -> Union[RandomForestClassifier, RandomForestRegressor]:
        # Use the provided parameters if they exist, otherwise use the instance dials
        params = parameters or self.model_dials

        common_params = {
            "n_estimators": params.n_estimators,
            "max_depth": params.max_depth,
            "min_samples_split": params.min_samples_split,
            "min_samples_leaf": params.min_samples_leaf,
            "max_features": params.max_features,
            "random_state": params.random_state,
            "bootstrap": params.bootstrap,
            "n_jobs": self.n_jobs,
        }

        if self.task_type == TaskType.REGRESSION:
            # Ensure a valid regression criterion is used
            # Default to squared_error if the params still have 'gini' from a copy-paste
            raw_crit = (
                params.criterion
                if params.criterion in get_args(RFRCriterion)
                else "squared_error"
            )

            crit = cast(RFRCriterion, raw_crit)

            return RandomForestRegressor(criterion=crit, **common_params)
        else:
            # Logic for Classification
            raw_crit = (
                params.criterion
                if params.criterion in get_args(RFCriterion)
                else "gini"
            )

            crit = cast(RFCriterion, raw_crit)

            return RandomForestClassifier(
                criterion=crit, class_weight=params.class_weight, **common_params
            )
