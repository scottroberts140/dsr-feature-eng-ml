"""
Models for machine learning implementations.

Provides different classifier models with hyperparameter tuning and evaluation.
"""

from dsr_feature_eng_ml.models.decision_tree import DecisionTree, DecisionTreeHyperParameters
from dsr_feature_eng_ml.models.random_forest import RandomForest, RandomForestHyperParameters
from dsr_feature_eng_ml.models.logistic_regression import LogisticRegression, LogisticRegressionHyperParameters
from dsr_feature_eng_ml.models.model_specification import ModelSpecification

__all__ = [
    "ModelSpecification",
    "DecisionTree",
    "DecisionTreeHyperParameters",
    "RandomForest",
    "RandomForestHyperParameters",
    "LogisticRegression",
    "LogisticRegressionHyperParameters",
]
