"""
Enumeration definitions for -specific classifications and configurations.
"""

from enum import Enum


class ModelGeneralization(Enum):
    """
    Enum to categorize model generalization quality based on F1 score gaps.

    This enum classifies how well a machine learning model generalizes from
    training data to validation data by examining the gap between training
    and validation F1 scores.

    Members:
        Undefined: Initial state or when generalization cannot be determined
        Good: Model generalizes well (small gap between train and validation F1)
        Acceptable: Model generalizes adequately (moderate gap)
        Overfitted: Model is overfitted (large gap between train and validation F1)

    The classification thresholds are typically defined by constants like
    DEFAULT_LARGE_GAP and DEFAULT_ACCEPTABLE_GAP to determine boundaries
    between these categories.
    """
    Undefined = 0
    Good = 1
    Acceptable = 2
    Overfitted = 3


class ModelType(Enum):
    """Enumeration of supported machine learning model types.

    Attributes:
        Undefined: Placeholder for uninitialized model type.
        Decision_Tree: Decision tree classifier model.
        Random_Forest: Random forest classifier model.
        Logistic_Regression: Logistic regression classifier model.
    """
    Undefined = 0
    Decision_Tree = 1
    Random_Forest = 2
    Logistic_Regression = 3


class ModelBalancing(Enum):
    """Enumeration of class balancing strategies for handling imbalanced datasets.

    Attributes:
        Undefined: Placeholder for uninitialized balancing strategy.
        Imbalance: No balancing applied, use original class distribution.
        Auto_Balanced: Use scikit-learn's 'balanced' class_weight parameter.
        Upsampled: Increase minority class samples to match majority class.
        Downsampled: Decrease majority class samples to match minority class.
    """
    Undefined = 0
    Imbalance = 1
    Auto_Balanced = 2
    Upsampled = 3
    Downsampled = 4


class ModelEvaluationMethod(Enum):
    """Enumeration of model evaluation and hyperparameter tuning methods.

    Attributes:
        Undefined: Placeholder for uninitialized evaluation method.
        Grid_Search: Exhaustive grid search over parameter space.
        Randomized_Search: Random sampling of parameter combinations.
        Confusion_Matrix: Evaluation using confusion matrix metrics.
        Upsampling: Evaluation with upsampled training data.
        Downsampling: Evaluation with downsampled training data.
    """
    Undefined = 0
    Grid_Search = 1
    Randomized_Search = 2
    Confusion_Matrix = 3
    Upsampling = 4
    Downsampling = 5


class ModelConfigurationSortOrder(Enum):
    """Enumeration of sorting options for model configuration results.

    Attributes:
        No_Sort: Maintain original order without sorting.
        F1_Score: Sort by F1 score metric.
        Model_Type: Sort by model type (Decision Tree, Random Forest, etc.).
        Model_Balancing: Sort by balancing strategy (Imbalance, Upsampled, etc.).
    """
    No_Sort = 0
    F1_Score = 1
    Model_Type = 2
    Model_Balancing = 3
