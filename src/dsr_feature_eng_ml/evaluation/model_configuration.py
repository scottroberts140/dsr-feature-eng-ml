from __future__ import annotations
from abc import ABC
import functools
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Union, Optional, Mapping
from dsr_feature_eng_ml.enums import ModelType, ModelBalancing, ModelEvaluationMethod, ModelGeneralization
from dsr_feature_eng_ml.evaluation.data_splits import DataSplits
from dsr_feature_eng_ml.constants import DEFAULT_LARGE_GAP, DEFAULT_ACCEPTABLE_GAP, F1_FORMAT, REPORT_WIDTH
from dsr_utils import format_text


@dataclass(frozen=True)
@functools.total_ordering
class ModelConfiguration:
    """Immutable container for model configuration and evaluation results.

    This frozen dataclass captures all parameters, settings, and performance 
    metrics for a specific model configuration. Once created, instances cannot 
    be modified, ensuring result integrity.

    Attributes:
        model_type (ModelType): Type of machine learning model used.
        model_balancing (ModelBalancing): Class balancing strategy applied.
        evaluation_method (ModelEvaluationMethod): Hyperparameter tuning method.
        params (dict): Model hyperparameters used for training.
        data_splits (DataSplits): Train/validation/test data splits.
        cv (int): Number of cross-validation folds.
        class_weight (Optional[Union[Mapping[str, float], str]]): Class weights or 'balanced'.
        scoring (str): Scoring metric (e.g., 'f1', 'accuracy').
        n_jobs (int): Number of parallel jobs for training.
        n_iter (int): Number of iterations for randomized search.
        features (list[str]): Feature columns used in the model.
        f1_score_cv (Optional[float]): F1 score from cross-validation (hyperparameter search).
        f1_score_train (Optional[float]): F1 score on training set.
        f1_score_valid (float): F1 score on validation set (primary comparison metric).
        model_generalization (ModelGeneralization): Generalization quality, calculated
            automatically based on the gap between training and validation F1 scores.
        acceptable_gap (float): Threshold for acceptable generalization gap (default: DEFAULT_ACCEPTABLE_GAP).
        large_gap (float): Threshold for large gap indicating overfitting (default: DEFAULT_LARGE_GAP).

    Example:
        >>> config = ModelConfiguration(
        ...     model_type=ModelType.Random_Forest,
        ...     model_balancing=ModelBalancing.Upsampled,
        ...     evaluation_method=ModelEvaluationMethod.Grid_Search,
        ...     params={'max_depth': 10, 'n_estimators': 100},
        ...     data_splits=splits,
        ...     cv=5,
        ...     class_weight=None,
        ...     scoring='f1',
        ...     n_jobs=-1,
        ...     n_iter=0,
        ...     features=['age', 'income'],
        ...     f1_score_cv=0.83,
        ...     f1_score_train=0.86,
        ...     f1_score_valid=0.85,
        ...     model_generalization=ModelGeneralization.Undefined,
        ...     acceptable_gap=DEFAULT_ACCEPTABLE_GAP,
        ...     large_gap=DEFAULT_LARGE_GAP,
        ... )
    """
    model_type: ModelType
    model_balancing: ModelBalancing
    evaluation_method: ModelEvaluationMethod
    params: dict
    features: list[str]
    data_splits: DataSplits
    f1_score_cv: Optional[float]
    f1_score_train: Optional[float]
    f1_score_valid: float
    cv: int
    class_weight: Optional[Union[
        Mapping[str, float],
        str
    ]]
    scoring: str
    n_jobs: int
    n_iter: int
    max_iter: int = 300
    model_generalization: ModelGeneralization = ModelGeneralization.Undefined
    acceptable_gap: float = DEFAULT_ACCEPTABLE_GAP
    large_gap: float = DEFAULT_LARGE_GAP

    def __post_init__(self) -> None:
        """Calculate model generalization after initialization.

        Uses object.__setattr__() to set the frozen attribute based on
        the gap between training and validation F1 scores.
        """
        # Calculate generalization status
        if self.f1_score_train is not None and self.f1_score_valid is not None:
            gap = self.f1_score_train - self.f1_score_valid

            if gap >= DEFAULT_LARGE_GAP:
                generalization = ModelGeneralization.Overfitted
            elif gap >= DEFAULT_ACCEPTABLE_GAP:
                generalization = ModelGeneralization.Acceptable
            else:
                generalization = ModelGeneralization.Good
        else:
            generalization = ModelGeneralization.Undefined

        # For frozen dataclasses, use object.__setattr__()
        object.__setattr__(self, 'model_generalization', generalization)

    @classmethod
    def empty(
        cls,
    ) -> ModelConfiguration:
        """Create an empty ModelConfiguration instance for initialization.

        Returns:
            ModelConfiguration: Empty configuration with default values.
        """
        return cls(
            model_type=ModelType.Undefined,
            model_balancing=ModelBalancing.Undefined,
            evaluation_method=ModelEvaluationMethod.Undefined,
            params={},
            data_splits=DataSplits.empty(),
            cv=0,
            class_weight=None,
            scoring='',
            n_jobs=0,
            n_iter=0,
            features=[],
            f1_score_cv=None,
            f1_score_train=None,
            f1_score_valid=0.0,
            acceptable_gap=DEFAULT_ACCEPTABLE_GAP,
            large_gap=DEFAULT_LARGE_GAP
        )

    def __hash__(self) -> int:
        """Make configuration hashable for use in sets/dicts."""
        return hash((
            self.model_type,
            self.model_balancing,
            self.evaluation_method,
            self.f1_score_valid,
            tuple(self.features)
        ))

    def __eq__(
            self,
            other: object
    ) -> bool:
        if not isinstance(other, ModelConfiguration):
            return NotImplemented
        return self.f1_score_valid == other.f1_score_valid

    def __lt__(
            self,
            other: object
    ) -> bool:
        if not isinstance(other, ModelConfiguration):
            return NotImplemented
        return self.f1_score_valid < other.f1_score_valid

    def info(self) -> str:
        """Display comprehensive information about this model configuration.

        Prints model type, balancing strategy, evaluation method, parameters,
        features used, and F1 score.
        """
        param_text = format_text(
            text=f'{self.params}',
            buffer_width=REPORT_WIDTH,
            prefix='',
            suffix='',
        )

        features_text = format_text(
            text=f'{self.features}',
            buffer_width=REPORT_WIDTH,
            prefix='',
            suffix='',
        )

        f1_cv_text = f'{self.f1_score_cv:{F1_FORMAT}}' if self.f1_score_cv is not None else 'N/A'
        f1_train_text = f'{self.f1_score_train:{F1_FORMAT}}' if self.f1_score_train is not None else 'N/A'
        f1_valid_text = f'{self.f1_score_valid:{F1_FORMAT}}' if self.f1_score_valid is not None else 'N/A'

        return f"""
Model Type: {self.model_type.name}
Model Balancing: {self.model_balancing.name}
Evaluation Method: {self.evaluation_method.name}
Parameters: {param_text}
Features: {features_text}
CV F1 Score:          {f1_cv_text}
Training F1 Score:    {f1_train_text}
Validation F1 Score:  {f1_valid_text}
Model Generalization: {self.model_generalization.name}
"""
