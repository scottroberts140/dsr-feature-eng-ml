"""
Enumeration definitions for -specific classifications and configurations.
"""

from __future__ import annotations
from enum import Enum, auto, Flag
from typing import Optional, Type, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from dsr_feature_eng_ml.models import ModelSpecification
import copy


class TaskType(Enum):
    """Supported machine learning task types."""

    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    UNKNOWN = "Unknown"
    # FUTURE: CLUSTERING = auto()

    @property
    def sort_order(self) -> int:
        return TaskTypeSortOrder[self.name].value


class TaskTypeSortOrder(Enum):
    REGRESSION = auto()
    CLASSIFICATION = auto()
    UNKNOWN = auto()
    # FUTURE: CLUSTERING = auto()


class ModelGeneralization(Enum):

    PENDING = "Pending"  # Audit hasn't run yet
    WELL_FIT = "Well-Fit"  # Gap <= acceptable_gap
    MARGINAL = "Marginal"  # acceptable_gap < Gap <= large_gap
    OVERFIT = "Overfit"  # Gap > large_gap
    UNDERFIT = "Underfit"  # (Optional: if score_train is too low)


class ModelType(Enum):
    # Regression
    LINEAR_REGRESSION = "Linear Regression"
    RANDOM_FOREST_REGRESSOR = "Random Forest Regressor"
    DECISION_TREE_REGRESSOR = "Decision Tree Regressor"
    RIDGE = "Ridge Regression"
    LASSO = "Lasso Regression"
    ELASTIC_NET = "Elastic Net"

    # Classification
    LOGISTIC_REGRESSION = "Logistic Regression"
    RANDOM_FOREST_CLASSIFIER = "Random Forest Classifier"
    DECISION_TREE_CLASSIFIER = "Decision Tree Classifier"

    # Unknown
    UNKNOWN = "Unknown"

    @property
    def abbrev(self) -> str:
        return ModelTypeAbbrev[self.name].value

    @property
    def tuning_multiplier(self) -> float:
        try:
            return ModelTypeTuningMultiplier[self.name].value
        except KeyError:
            return 3.0

    @property
    def task_type(self) -> TaskType:
        return ModelTypeTaskType[self.name].value

    @property
    def model_class(self) -> Optional[Type["ModelSpecification"]]:
        from dsr_feature_eng_ml.models.decision_tree import DecisionTree
        from dsr_feature_eng_ml.models.elastic_net_regression import (
            ElasticNetRegression,
        )
        from dsr_feature_eng_ml.models.lasso_regression import LassoRegression
        from dsr_feature_eng_ml.models.linear_regression import LinearRegression
        from dsr_feature_eng_ml.models.logistic_regression import LogisticRegression
        from dsr_feature_eng_ml.models.random_forest import RandomForest
        from dsr_feature_eng_ml.models.ridge_regression import RidgeRegression

        match self:
            case ModelType.LINEAR_REGRESSION:
                return LinearRegression
            case ModelType.RANDOM_FOREST_REGRESSOR:
                return RandomForest
            case ModelType.DECISION_TREE_REGRESSOR:
                return DecisionTree
            case ModelType.RIDGE:
                return RidgeRegression
            case ModelType.LASSO:
                return LassoRegression
            case ModelType.ELASTIC_NET:
                return ElasticNetRegression
            case ModelType.LOGISTIC_REGRESSION:
                return LogisticRegression
            case ModelType.RANDOM_FOREST_CLASSIFIER:
                return RandomForest
            case ModelType.DECISION_TREE_CLASSIFIER:
                return DecisionTree

        return None


class ModelTypeAbbrev(Enum):
    # Regression
    LINEAR_REGRESSION = "LIN"
    RANDOM_FOREST_REGRESSOR = "RFR"
    DECISION_TREE_REGRESSOR = "DTR"
    RIDGE = "RGE"
    LASSO = "LAS"
    ELASTIC_NET = "ENT"

    # Classification
    LOGISTIC_REGRESSION = "LOG"
    RANDOM_FOREST_CLASSIFIER = "RFC"
    DECISION_TREE_CLASSIFIER = "DTC"

    # Unknown
    UNKNOWN = "UNK"


class ModelTypeTaskType(Enum):
    # Regression
    LINEAR_REGRESSION = TaskType.REGRESSION
    RANDOM_FOREST_REGRESSOR = TaskType.REGRESSION
    DECISION_TREE_REGRESSOR = TaskType.REGRESSION
    RIDGE = TaskType.REGRESSION
    LASSO = TaskType.REGRESSION
    ELASTIC_NET = TaskType.REGRESSION

    # Classification
    LOGISTIC_REGRESSION = TaskType.CLASSIFICATION
    RANDOM_FOREST_CLASSIFIER = TaskType.CLASSIFICATION
    DECISION_TREE_CLASSIFIER = TaskType.CLASSIFICATION

    # Unknown
    UNKNOWN = TaskType.UNKNOWN


class ModelTypeTuningMultiplier(Enum):
    # High Risk (Tree/Ensemble)
    RANDOM_FOREST_REGRESSOR = 15.0
    RANDOM_FOREST_CLASSIFIER = 18.0  # Higher due to class-prob tracking
    XGB_CLASSIFIER = 12.0

    # Extreme Risk (Memory-based)
    K_NEIGHBORS_CLASSIFIER = 25.0

    # Low Risk (Linear/Parametric)
    LOGISTIC_REGRESSION = 3.0
    LINEAR_SVC = 3.0
    RIDGE_CLASSIFIER = 3.0

    # Unknown
    UNKNOWN = 3.0


class ModelEnumSortOrder(Flag):
    NAME = auto()
    VALUE = auto()
    ABBREV = auto()
    TASK_TYPE_NAME = auto()
    TASK_TYPE_ABBREV = auto()


@dataclass
class ModelTypeData:
    rec_type: ModelTypeDataRecType
    model_type: ModelType
    name: str
    value: str
    abbrev: str
    tuning_multiplier: float
    task_type: TaskType

    @classmethod
    def create_header_from_item(cls, item: ModelTypeData) -> ModelTypeData:
        header = copy.copy(item)
        header.rec_type = ModelTypeDataRecType.HEADER
        return header

    @classmethod
    def get_list(
        cls, sort_order: ModelEnumSortOrder, include_task_type_headers: bool = False
    ) -> list[ModelTypeData]:
        model_types = list(ModelType)
        model_list: list[ModelTypeData] = [
            cls(
                rec_type=ModelTypeDataRecType.DATA,
                model_type=mt,
                name=mt.name,
                value=mt.value,
                abbrev=mt.abbrev,
                tuning_multiplier=mt.tuning_multiplier,
                task_type=mt.task_type,
            )
            for mt in model_types
        ]

        match sort_order:
            case ModelEnumSortOrder.NAME:
                mtd_list = sorted(model_list, key=lambda mtd: (mtd.name))
            case ModelEnumSortOrder.VALUE:
                mtd_list = sorted(model_list, key=lambda mtd: (mtd.value))
            case ModelEnumSortOrder.ABBREV:
                mtd_list = sorted(model_list, key=lambda mtd: (mtd.abbrev))
            case ModelEnumSortOrder.TASK_TYPE_NAME:
                mtd_list = sorted(
                    model_list, key=lambda mtd: (mtd.task_type.sort_order, mtd.name)
                )
            case ModelEnumSortOrder.TASK_TYPE_ABBREV:
                mtd_list = sorted(
                    model_list, key=lambda mtd: (mtd.task_type.sort_order, mtd.abbrev)
                )
            case _:
                mtd_list = sorted(model_list, key=lambda mtd: (mtd.name))

        if include_task_type_headers:
            current_task_type = ""
            final_list: list[ModelTypeData] = []

            for _, mtd in enumerate(mtd_list):
                if mtd.task_type.value != current_task_type:
                    final_list.append(cls.create_header_from_item(mtd))
                    current_task_type = mtd.task_type.value

                final_list.append(mtd)
        else:
            final_list = mtd_list

        return final_list


class ModelTypeDataRecType(Enum):
    HEADER = auto()
    DATA = auto()


class BalancingStrategy(Enum):
    NONE = "None"
    UNBALANCED = "Unbalanced"
    WEIGHTED = "Weighted"
    OVERSAMPLED = "Oversampled"
    UNDERSAMPLED = "Undersampled"

    @property
    def abbrev(self) -> str:
        return BalancingStrategyAbbrev[self.name].value


class BalancingStrategyAbbrev(Enum):
    NONE = "N"
    UNBALANCED = "X"
    WEIGHTED = "W"
    OVERSAMPLED = "O"
    UNDERSAMPLED = "U"


class OptimizationStrategy(Enum):
    MANUAL = "Manual"
    GRID_SEARCH = "Grid Search"
    RANDOM_SEARCH = "Random Search"


class ModelConfigurationSortOrder(Enum):
    NONE = "None"
    F1 = "F1"
    MODEL_TYPE = "Model Type"
    BALANCING_STRATEGY = "Balancing Strategy"


class ScoringMetric(Enum):
    """Supported scoring metric keys for scikit-learn workflows.

    Includes common classification metrics (F1, accuracy, precision, recall)
    and regression metrics (MAE, MSE, R2). Values match the strings expected by
    scikit-learn's ``scoring`` parameter, and ``get_valid_metrics`` returns the
    allowed options for a given ``TaskType``.
    """

    # Classification
    F1 = "f1"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"

    # Regression
    MAE = "neg_mean_absolute_error"
    MSE = "neg_mean_squared_error"
    R2 = "r2"

    # Unknown
    UNKNOWN = "Unknown"

    @classmethod
    def get_valid_metrics(cls, task_type: TaskType) -> list[ScoringMetric]:
        if task_type == TaskType.CLASSIFICATION:
            return [
                cls.F1,
                cls.ACCURACY,
                cls.PRECISION,
                cls.RECALL,
            ]
        return [cls.MAE, cls.MSE, cls.R2]


class PlotFileName(Enum):
    CV_VS_FINAL = "cv_vs_final"
    TOP_FEATURES = "top_features"
    CUM_IMPORTANCE = "cum_importance"
