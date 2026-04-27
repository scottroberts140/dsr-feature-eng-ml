"""Enumeration definitions for dsr_feature_eng_ml classifications and configs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import TYPE_CHECKING, Any, Optional, Type

if TYPE_CHECKING:
    from dsr_feature_eng_ml.models import ModelSpecification

import copy


class TaskType(StrEnum):
    """
    Supported machine learning task types.

    Members
    -------
    REGRESSION : str
        Predicting continuous numerical values.
    CLASSIFICATION : str
        Predicting discrete categorical labels.
    UNKNOWN : str
        Fallback for uninitialized or invalid task types.
    """

    REGRESSION = "Regression"
    CLASSIFICATION = "Classification"
    UNKNOWN = "Unknown"
    # FUTURE: CLUSTERING = auto()

    @property
    def sort_order(self) -> int:
        """
        Retrieves the integer priority for sorting task types.

        Returns
        -------
        int
            The sort order value defined in TaskTypeSortOrder.
        """
        return TaskTypeSortOrder[self.name].value


class TaskTypeSortOrder(Enum):
    """
    Priority mapping for sorting TaskTypes in reports and UI lists.
    """

    REGRESSION = auto()
    CLASSIFICATION = auto()
    UNKNOWN = auto()
    # FUTURE: CLUSTERING = auto()


MODEL_METADATA: dict[str, dict[str, Any]] = {
    "LINEAR_REGRESSION": {
        "abbrev": "LIN",
        "task": TaskType.REGRESSION,
        "multiplier": 3.0,
    },
    "RANDOM_FOREST_REGRESSOR": {
        "abbrev": "RFR",
        "task": TaskType.REGRESSION,
        "multiplier": 15.0,
    },
    "DECISION_TREE_REGRESSOR": {
        "abbrev": "DTR",
        "task": TaskType.REGRESSION,
        "multiplier": 3.0,
    },
    "RIDGE": {"abbrev": "RGE", "task": TaskType.REGRESSION, "multiplier": 3.0},
    "LASSO": {"abbrev": "LAS", "task": TaskType.REGRESSION, "multiplier": 3.0},
    "ELASTIC_NET": {"abbrev": "ENT", "task": TaskType.REGRESSION, "multiplier": 3.0},
    "LOGISTIC_REGRESSION": {
        "abbrev": "LOG",
        "task": TaskType.CLASSIFICATION,
        "multiplier": 3.0,
    },
    "RANDOM_FOREST_CLASSIFIER": {
        "abbrev": "RFC",
        "task": TaskType.CLASSIFICATION,
        "multiplier": 18.0,
    },
    "DECISION_TREE_CLASSIFIER": {
        "abbrev": "DTC",
        "task": TaskType.CLASSIFICATION,
        "multiplier": 3.0,
    },
    "XGB_CLASSIFIER": {
        "abbrev": "XGB",
        "task": TaskType.CLASSIFICATION,
        "multiplier": 12.0,
    },
    "LINEAR_SVC": {"abbrev": "SVC", "task": TaskType.CLASSIFICATION, "multiplier": 3.0},
    "RIDGE_CLASSIFIER": {
        "abbrev": "RGC",
        "task": TaskType.CLASSIFICATION,
        "multiplier": 3.0,
    },
    "K_NEIGHBORS_CLASSIFIER": {
        "abbrev": "KNN",
        "task": TaskType.CLASSIFICATION,
        "multiplier": 25.0,
    },
    "UNKNOWN": {"abbrev": "UNK", "task": TaskType.UNKNOWN, "multiplier": 3.0},
}


class ModelGeneralization(StrEnum):
    """
    Classification of model performance based on the train-test gap.

    Members
    -------
    PENDING : str
        Audit hasn't run yet.
    WELL_FIT : str
        Gap between train and test scores is within acceptable limits.
    MARGINAL : str
        Gap is slightly elevated but potentially acceptable.
    OVERFIT : str
        Gap indicates high variance; model is learning noise.
    UNDERFIT : str
        Model performance is poor on both training and test data.
    """

    PENDING = "Pending"  # Audit hasn't run yet
    WELL_FIT = "Well-Fit"  # Gap <= acceptable_gap
    MARGINAL = "Marginal"  # acceptable_gap < Gap <= large_gap
    OVERFIT = "Overfit"  # Gap > large_gap
    UNDERFIT = "Underfit"  # (Optional: if score_train is too low)


class ModelType(Enum):
    """
    Supported machine learning models for regression and classification tasks.

    Members
    -------
    LINEAR_REGRESSION : str
        Ordinary Least Squares Linear Regression.
    RANDOM_FOREST_REGRESSOR : str
        Random Forest Meta-Estimator for regression.
    DECISION_TREE_REGRESSOR : str
        Individual Decision Tree for regression.
    RIDGE : str
        L2-regularized Linear Regression.
    LASSO : str
        L1-regularized Linear Regression.
    ELASTIC_NET : str
        Combined L1 and L2 regularized Linear Regression.
    LOGISTIC_REGRESSION : str
        Logistic Regression for binary or multiclass classification.
    RANDOM_FOREST_CLASSIFIER : str
        Random Forest Meta-Estimator for classification.
    DECISION_TREE_CLASSIFIER : str
        Individual Decision Tree for classification.
    XGB_CLASSIFIER : str
        Extreme Gradient Boosting for classification.
    LINEAR_SVC : str
        Linear Support Vector Classification.
    RIDGE_CLASSIFIER : str
        Ridge Regression adapted for classification tasks.
    K_NEIGHBORS_CLASSIFIER : str
        K-Nearest Neighbors classification.
    UNKNOWN : str
        Fallback for uninitialized or unsupported model types.
    """

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
    XGB_CLASSIFIER = "XGBoost Classifier"
    LINEAR_SVC = "Linear SVC"
    RIDGE_CLASSIFIER = "Ridge Classifier"
    K_NEIGHBORS_CLASSIFIER = "K-Neighbors Classifier"

    # Unknown
    UNKNOWN = "Unknown"

    # Internal Metadata Mapping: Consolidates parallel enums into a single source of truth
    _ignore_ = ["MODEL_METADATA"]

    @property
    def abbrev(self) -> str:
        """
        Retrieves the standardized short-form abbreviation for the model.

        Returns
        -------
        str
            A 3-character uppercase abbreviation.
        """
        return MODEL_METADATA.get(self.name, {}).get("abbrev", "UNK")

    @property
    def tuning_multiplier(self) -> float:
        """
        Retrieves the difficulty weight for hyperparameter tuning.

        Returns
        -------
        float
            Factor used to estimate computational budget for this model.
        """
        return MODEL_METADATA.get(self.name, {}).get("multiplier", 3.0)

    @property
    def task_type(self) -> TaskType:
        """
        Identifies whether the model is for Regression or Classification.

        Returns
        -------
        TaskType
            The task category associated with this model.
        """
        return MODEL_METADATA.get(self.name, {}).get("task", TaskType.UNKNOWN)

    @property
    def model_class(self) -> Optional[Type["ModelSpecification"]]:
        """
        Retrieves the corresponding ModelSpecification class for instantiation.

        Returns
        -------
        Type[ModelSpecification] or None
            The uninitialized class reference for the model wrapper.
        """
        # (The match statement implementation remains the same,
        # returning None for unimplemented models)
        from dsr_feature_eng_ml.models.decision_tree import (
            DecisionTreeClassifierModel,
            DecisionTreeRegressorModel,
        )
        from dsr_feature_eng_ml.models.elastic_net_regression import (
            ElasticNetRegression,
        )
        from dsr_feature_eng_ml.models.k_neighbors_classifier import (
            KNeighborsClassifierModel,
        )
        from dsr_feature_eng_ml.models.lasso_regression import LassoRegression
        from dsr_feature_eng_ml.models.linear_regression import LinearRegression
        from dsr_feature_eng_ml.models.linear_svc import LinearSVCModel
        from dsr_feature_eng_ml.models.logistic_regression import LogisticRegression
        from dsr_feature_eng_ml.models.random_forest import (
            RandomForestClassifierModel,
            RandomForestRegressorModel,
        )
        from dsr_feature_eng_ml.models.ridge_classifier import RidgeClassifierModel
        from dsr_feature_eng_ml.models.ridge_regression import RidgeRegression
        from dsr_feature_eng_ml.models.xgboost_classifier import XGBClassifierModel

        match self:
            case ModelType.LINEAR_REGRESSION:
                return LinearRegression
            case ModelType.RANDOM_FOREST_REGRESSOR:
                return RandomForestRegressorModel
            case ModelType.DECISION_TREE_REGRESSOR:
                return DecisionTreeRegressorModel
            case ModelType.RIDGE:
                return RidgeRegression
            case ModelType.LASSO:
                return LassoRegression
            case ModelType.ELASTIC_NET:
                return ElasticNetRegression
            case ModelType.LOGISTIC_REGRESSION:
                return LogisticRegression
            case ModelType.RANDOM_FOREST_CLASSIFIER:
                return RandomForestClassifierModel
            case ModelType.DECISION_TREE_CLASSIFIER:
                return DecisionTreeClassifierModel
            case ModelType.XGB_CLASSIFIER:
                return XGBClassifierModel
            case ModelType.RIDGE_CLASSIFIER:
                return RidgeClassifierModel
            case ModelType.LINEAR_SVC:
                return LinearSVCModel
            case ModelType.K_NEIGHBORS_CLASSIFIER:
                return KNeighborsClassifierModel

        return None


class ModelEnumSortOrder(Enum):
    """Determines the primary key used to sort model data lists."""

    NAME = auto()
    VALUE = auto()
    ABBREV = auto()
    TASK_TYPE_NAME = auto()
    TASK_TYPE_ABBREV = auto()


@dataclass
class ModelTypeData:
    """
    Flat metadata representation of a ModelType for reporting and UI usage.

    Attributes
    ----------
    rec_type : ModelTypeDataRecType
        Determines if the instance is a data row or a header.
    model_type : ModelType
        The reference to the source model enum.
    name : str
        The internal name of the model type.
    value : str
        The human-readable label of the model.
    abbrev : str
        The 3-character abbreviation.
    tuning_multiplier : float
        Difficulty weight for hyperparameter search.
    task_type : TaskType
        The task category (Regression/Classification).
    """

    rec_type: ModelTypeDataRecType
    model_type: ModelType
    name: str
    value: str
    abbrev: str
    tuning_multiplier: float
    task_type: TaskType

    @classmethod
    def create_header_from_item(cls, item: ModelTypeData) -> ModelTypeData:
        """
        Creates a category header based on an existing data item.

        Parameters
        ----------
        item : ModelTypeData
            The data item whose TaskType will be used for the header.

        Returns
        -------
        ModelTypeData
            A copy of the item with the rec_type set to HEADER.
        """
        header = copy.copy(item)
        header.rec_type = ModelTypeDataRecType.HEADER
        return header

    @classmethod
    def get_list(
        cls, sort_order: ModelEnumSortOrder, include_task_type_headers: bool = False
    ) -> list[ModelTypeData]:
        """
        Retrieves a sorted list of model metadata.

        Parameters
        ----------
        sort_order : ModelEnumSortOrder
            The criteria used for sorting.
        include_task_type_headers : bool, default False
            If True, inserts category headers whenever the TaskType changes.

        Returns
        -------
        list of ModelTypeData
            The sorted list of metadata objects.
        """
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
                mtd_list = sorted(model_list, key=lambda mtd: mtd.name)
            case ModelEnumSortOrder.VALUE:
                mtd_list = sorted(model_list, key=lambda mtd: mtd.value)
            case ModelEnumSortOrder.ABBREV:
                mtd_list = sorted(model_list, key=lambda mtd: mtd.abbrev)
            case ModelEnumSortOrder.TASK_TYPE_NAME:
                mtd_list = sorted(
                    model_list, key=lambda mtd: (mtd.task_type.sort_order, mtd.name)
                )
            case ModelEnumSortOrder.TASK_TYPE_ABBREV:
                mtd_list = sorted(
                    model_list, key=lambda mtd: (mtd.task_type.sort_order, mtd.abbrev)
                )
            case _:
                mtd_list = sorted(model_list, key=lambda mtd: mtd.name)

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
    """Differentiates between metadata rows and category headers."""

    HEADER = auto()
    DATA = auto()


BALANCING_METADATA: dict[str, str] = {
    "NONE": "N",
    "UNBALANCED": "X",
    "WEIGHTED": "W",
    "OVERSAMPLED": "O",
    "UNDERSAMPLED": "U",
}


class BalancingStrategy(StrEnum):
    """
    Strategies for addressing class imbalance in classification tasks.

    Members
    -------
    NONE : str
        No balancing applied.
    UNBALANCED : str
        Explicitly identifies the dataset as remaining unbalanced.
    WEIGHTED : str
        Adjusts model weights inversely proportional to class frequencies.
    OVERSAMPLED : str
        Increases the count of the minority class (e.g., via SMOTE).
    UNDERSAMPLED : str
        Reduces the count of the majority class.
    """

    NONE = "None"
    UNBALANCED = "Unbalanced"
    WEIGHTED = "Weighted"
    OVERSAMPLED = "Oversampled"
    UNDERSAMPLED = "Undersampled"

    @property
    def abbrev(self) -> str:
        """
        Retrieves the standardized short-form abbreviation for the strategy.

        Returns
        -------
        str
            A single-character uppercase abbreviation (e.g., 'W', 'O').
        """
        return BALANCING_METADATA.get(self.name, "N")


class OptimizationStrategy(StrEnum):
    """Methodologies for hyperparameter tuning."""

    MANUAL = "Manual"
    GRID_SEARCH = "Grid Search"
    RANDOM_SEARCH = "Random Search"


class ModelConfigurationSortOrder(StrEnum):
    """Sort keys for model configuration ranking."""

    NONE = "None"
    F1 = "F1"
    MODEL_TYPE = "Model Type"
    BALANCING_STRATEGY = "Balancing Strategy"


class ScoringMetric(Enum):
    """
    Supported scoring metric keys for scikit-learn workflows.

    Includes common classification metrics (F1, accuracy, precision, recall)
    and regression metrics (MAE, MSE, R2). Values match the strings expected by
    scikit-learn's ``scoring`` parameter.
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
        """
        Retrieves the allowed scoring metrics for a specific task type.

        Parameters
        ----------
        task_type : TaskType
            The type of machine learning task (Regression or Classification).

        Returns
        -------
        list of ScoringMetric
            A list of metrics applicable to the provided task type.
        """
        if task_type == TaskType.CLASSIFICATION:
            return [
                cls.F1,
                cls.ACCURACY,
                cls.PRECISION,
                cls.RECALL,
            ]
        return [cls.MAE, cls.MSE, cls.R2]


class PlotFileName(StrEnum):
    """Standard filenames for visualization outputs."""

    CV_VS_FINAL = "cv_vs_final"
    TOP_FEATURES = "top_features"
    CUM_IMPORTANCE = "cum_importance"
