"""
dsr_feature_eng_ml: Machine learning-specific feature engineering utilities.

Provides enums for model configuration, evaluation methods, and data balancing strategies,
along with specialized ML tools for feature engineering workflows.
"""

from dsr_feature_eng_ml.constants import (
    F1_FORMAT,
    REPORT_WIDTH,
    DEFAULT_VIABLE_F1_GAP,
    DEFAULT_ACCEPTABLE_GAP,
    DEFAULT_LARGE_GAP,
)
from dsr_feature_eng_ml.enums import (
    ModelGeneralization,
    ModelType,
    ModelBalancing,
    ModelEvaluationMethod,
    ModelConfigurationSortOrder,
)
from dsr_feature_eng_ml.evaluation import (
    DataSplits,
    ValidationResults,
    ModelFeatureImportance,
    ModelConfiguration,
    ModelEvaluation,
    ModelResults,
    BestModelResults,
)
from dsr_feature_eng_ml.models import (
    ModelSpecification,
    DecisionTree,
    DecisionTreeHyperParameters,
    RandomForest,
    RandomForestHyperParameters,
    LogisticRegression,
    LogisticRegressionHyperParameters,
)

__all__ = [
    "F1_FORMAT",
    "REPORT_WIDTH",
    "DEFAULT_VIABLE_F1_GAP",
    "DEFAULT_ACCEPTABLE_GAP",
    "DEFAULT_LARGE_GAP",
    "ModelGeneralization",
    "ModelType",
    "ModelBalancing",
    "ModelEvaluationMethod",
    "ModelConfigurationSortOrder",
    "DataSplits",
    "ValidationResults",
    "ModelFeatureImportance",
    "ModelConfiguration",
    "ModelEvaluation",
    "ModelResults",
    "BestModelResults",
    "ModelSpecification",
    "DecisionTree",
    "DecisionTreeHyperParameters",
    "RandomForest",
    "RandomForestHyperParameters",
    "LogisticRegression",
    "LogisticRegressionHyperParameters",
]

__version__ = "0.0.1"
