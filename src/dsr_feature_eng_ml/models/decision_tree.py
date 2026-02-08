from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Union,
    Optional,
    Literal,
    get_args,
    cast,
    TYPE_CHECKING,
    Callable,
    Type,
)
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dsr_utils import format_label_value_pairs

if TYPE_CHECKING:
    from dsr_feature_eng_ml.models.model_specification import (
        ScikitModel,
        ProbabilisticClassifier,
    )

# Decision Tree Specific Literals
DTCriterion = Literal["gini", "entropy", "log_loss"]
DTRCriterion = Literal["squared_error", "friedman_mse", "absolute_error", "poisson"]

TreeEstimator = Union[DecisionTreeClassifier, DecisionTreeRegressor]
TreeClass = Union[Type[DecisionTreeClassifier], Type[DecisionTreeRegressor]]


@dataclass(frozen=True)
class DecisionTreeParams(ModelParams):
    criterion: Literal[
        "gini",
        "entropy",
        "log_loss",
        "squared_error",
        "friedman_mse",
        "absolute_error",
        "poisson",
    ] = "gini"
    splitter: Literal["best", "random"] = "best"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[Union[int, float, Literal["sqrt", "log2"]]] = None
    random_state: Optional[int] = None
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    class_weight: Optional[Union[dict[str, float], str]] = None
    ccp_alpha: float = 0.0
    task_type: TaskType = TaskType.CLASSIFICATION
    scoring: ScoringMetric = ScoringMetric.F1

    @classmethod
    def create_default(
        cls, task_type: TaskType, scoring: ScoringMetric, random_state: int, **kwargs
    ) -> "DecisionTreeParams":
        if task_type == TaskType.REGRESSION:
            defaults = {
                "task_type": task_type,
                "criterion": "squared_error",  # Regression specific
                "max_features": None,  # Default for DecisionTreeRegressor
                "random_state": random_state,
                "class_weight": None,  # Not applicable to regression
                "scoring": scoring,
                **kwargs,
            }
        else:
            defaults = {
                "task_type": task_type,
                "criterion": "gini",  # Classification specific
                "max_features": None,  # Scikit-learn default is None (all)
                "random_state": random_state,
                "scoring": scoring,
                **kwargs,
            }
        return cls(**defaults)

    def __post_init__(self):
        """Validates that the criterion matches the task type."""
        if self.task_type == TaskType.CLASSIFICATION:
            if self.criterion not in get_args(DTCriterion):
                raise ValueError(
                    f"Invalid criterion '{self.criterion}' for Classification. "
                    f"Expected one of {get_args(DTCriterion)}"
                )
        else:
            if self.criterion not in get_args(DTRCriterion):
                raise ValueError(
                    f"Invalid criterion '{self.criterion}' for Regression. "
                    f"Expected one of {get_args(DTRCriterion)}"
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
            ("Task Type", f"{self.task_type}"),
            ("Scoring", f"{self.scoring}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list]:
        """Generate a standard hyperparameter search grid for DecisionTree.

        Provides predefined parameter combinations for hyperparameter tuning.
        The "narrow" mode focuses on core tree depth and leaf size levers,
        while the expanded mode includes additional fine-tuning parameters.

        Args:
            narrow (bool, optional): If True (default), returns core parameters
                (max_depth, min_samples_leaf). If False, includes additional
                parameters (min_samples_split, criterion, class_weight).

        Returns:
            dict[str, list]: Parameter grid with keys mapping to lists of values.
                - narrow=True: max_depth [None, 5, 10, 15, 20],
                              min_samples_leaf [1, 2, 5, 10]
                - narrow=False: Above plus min_samples_split [2, 5, 10],
                               criterion ["gini", "entropy"],
                               class_weight [None, "balanced"]

        Example:
            >>> narrow_grid = DecisionTreeParams.get_standard_search_grid()
            >>> expanded_grid = DecisionTreeParams.get_standard_search_grid(narrow=False)
        """
        # The "Core" levers always included
        grid = {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 5, 10],
        }

        if not narrow:
            # Add the fine-tuning levers
            grid.update(
                {
                    "min_samples_split": [2, 5, 10],
                    "criterion": ["gini", "entropy"],
                    "class_weight": [None, "balanced"],
                }
            )

        return grid


class DecisionTree(
    ModelSpecification[
        DecisionTreeParams, Union[DecisionTreeClassifier, DecisionTreeRegressor]
    ]
):
    def get_estimator_class(
        self,
    ) -> Type[Union[DecisionTreeClassifier, DecisionTreeRegressor]]:
        if self.task_type == TaskType.CLASSIFICATION:
            return DecisionTreeClassifier
        return DecisionTreeRegressor

    params_class = DecisionTreeParams

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def scoring(self) -> ScoringMetric:
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric):
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

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy,
        params: Optional[DecisionTreeParams] = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        scoring: ScoringMetric = ScoringMetric.F1,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        # Use provided params or fall back to defaults
        # We ensure the data's random_state is used if not specified in params
        if params is None:
            params = DecisionTreeParams.create_default(
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

        if self.task_type == TaskType.CLASSIFICATION:
            self._model_type = ModelType.DECISION_TREE_CLASSIFIER
        else:
            self._model_type = ModelType.DECISION_TREE_REGRESSOR

        self.optimization_strategy = optimization_strategy
        self.estimator = self.create_estimator()

    def create_estimator(
        self, parameters: Optional[DecisionTreeParams] = None
    ) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        # Use the provided parameters if they exist, otherwise use the instance dials
        params = parameters or self.model_dials

        # Common parameters shared by both Regressor and Classifier
        # Note: DecisionTree does NOT have n_jobs
        common_params = {
            "splitter": params.splitter,
            "max_depth": params.max_depth,
            "min_samples_split": params.min_samples_split,
            "min_samples_leaf": params.min_samples_leaf,
            "min_weight_fraction_leaf": params.min_weight_fraction_leaf,
            "max_features": params.max_features,
            "random_state": params.random_state,
            "max_leaf_nodes": params.max_leaf_nodes,
            "min_impurity_decrease": params.min_impurity_decrease,
            "ccp_alpha": params.ccp_alpha,
        }

        if self.task_type == TaskType.REGRESSION:
            # Fallback logic for Regression Criterion
            raw_crit = (
                params.criterion
                if params.criterion in get_args(DTRCriterion)
                else "squared_error"
            )
            crit = cast(DTRCriterion, raw_crit)

            return DecisionTreeRegressor(criterion=crit, **common_params)
        else:
            # Fallback logic for Classification Criterion
            raw_crit = (
                params.criterion
                if params.criterion in get_args(DTCriterion)
                else "gini"
            )
            crit = cast(DTCriterion, raw_crit)

            return DecisionTreeClassifier(
                criterion=crit, class_weight=params.class_weight, **common_params
            )
