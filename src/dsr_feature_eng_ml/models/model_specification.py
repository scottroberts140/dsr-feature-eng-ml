from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Union, Optional, cast, Mapping, Type, Any, Final
from dsr_feature_eng_ml.enums import ModelType, ModelBalancing, ModelEvaluationMethod
from dsr_feature_eng_ml.evaluation.data_splits import DataSplits
from dsr_feature_eng_ml.evaluation.model_configuration import ModelConfiguration
from dsr_feature_eng_ml.constants import DEFAULT_LARGE_GAP, DEFAULT_ACCEPTABLE_GAP, F1_FORMAT
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


class ModelSpecification(ABC):
    """Abstract base class for model specifications with common training parameters.

    Provides shared functionality for model training, validation prediction, and
    performance evaluation. Cannot be instantiated directly - must be subclassed
    by specific model types (Decision Tree, Random Forest, Logistic Regression).

    This class uses managed mutability: the predicted_valid attribute is modified
    by the calc_predicted_valid() method during model evaluation, but this is
    controlled through the class interface rather than external manipulation.

    Attributes:
        data_splits (DataSplits): Train/validation/test data splits.
        cv (int): Number of cross-validation folds.
        class_weight (Optional[Union[Mapping[str, float], str]]): Class weights.
        scoring (str): Scoring metric for model evaluation.
        n_jobs (int): Number of parallel jobs (-1 for all CPUs).
        n_iter (int): Number of iterations for randomized search.
        predicted_valid (pd.Series): Validation set predictions (modified by calc_predicted_valid).

    Example:
        >>> # Cannot instantiate directly - use subclasses
        >>> dtree = DecisionTree(
        ...     data_splits=splits,
        ...     cv=5,
        ...     param_grid={'max_depth': [10, 20]},
        ...     class_weight='balanced'
        ... )

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses inherit shared functionality while implementing model-specific behavior.
    """

    def __init__(
        self,
        data_splits: DataSplits,
        cv: int,
        class_weight: Optional[Union[
            Mapping[str, float],
            str
        ]],
        scoring: str,
        n_jobs: int,
        n_iter: int,
        acceptable_gap: float = DEFAULT_ACCEPTABLE_GAP,
        large_gap: float = DEFAULT_LARGE_GAP,
    ):
        self.data_splits = data_splits
        self.cv = cv
        self.class_weight = class_weight
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.acceptable_gap = acceptable_gap
        self.large_gap = large_gap
        self.predicted_valid = pd.Series(dtype=float)

    @property
    def random_state(self) -> int:
        return self.data_splits.random_state

    def calc_predicted_valid(
            self,
            model,
            include_in_report: bool = False,
            show_plot: bool = False
    ) -> str:
        """Generate predictions on validation set and analyze target frequency.

        Args:
            model: Trained model to use for predictions.
            include_in_report (bool): Whether to print target frequency statistics.
            show_plot (bool): Whether to display a bar plot of target frequency.

        Note:
            Updates the predicted_valid attribute with validation predictions.
        """
        report_text = ''
        self.predicted_valid = pd.Series(
            model.predict(self.data_splits.features_valid))
        target_frequency = self.predicted_valid.value_counts(normalize=True)

        if include_in_report:
            report_text = f'''

Target Frequency: {target_frequency}
'''

        if show_plot:
            target_frequency.plot(
                kind='bar',
                y=self.data_splits.target_column,
                ylabel='Frequency',
                title=f'Target Frequency [{self.data_splits.target_column}] (Validation Set)'
            )

        return report_text

    def calc_target_frequency(
            self,
            model_type: ModelType,
            model_balancing: ModelBalancing,
            params: dict,
            df: pd.Series,
            include_in_report: bool = False
    ) -> tuple[ModelConfiguration, str]:
        """Calculate confusion matrix metrics and return model configuration.

        Args:
            model_type (ModelType): Type of model being evaluated.
            model_balancing (ModelBalancing): Balancing strategy used.
            params (dict): Model hyperparameters.
            df (pd.Series): True target values for comparison.
            include_in_report (bool): Whether to print confusion matrix and metrics.

        Returns:
            ModelConfiguration: Configuration object with evaluation results.

        Note:
            Calculates true positive/negative rates, recall, precision, and F1 score.
        """
        report_text = ''
        cm = confusion_matrix(
            df,
            self.predicted_valid
        )

        tp = cm[0, 0]
        tn = cm[1, 1]
        total_positives = cm[0].sum()
        total_negatives = cm[1].sum()
        recall_result = recall_score(df, self.predicted_valid)
        precision_result = precision_score(df, self.predicted_valid)
        f1_result = cast(float, f1_score(df, self.predicted_valid))

        if include_in_report:
            report_text = f'''
Confusion Matrix: 
{cm}

Total Positives:  {total_positives}
Total Negatives:  {total_negatives}

True Positive:    {tp / total_positives:.2%}
True Negative:    {tn / total_negatives:.2%}

Recall Score:     {recall_result:{F1_FORMAT}}
Precision Score:  {precision_result:{F1_FORMAT}}
F1 Score:         {f1_result:{F1_FORMAT}}

'''

        # Get F1 scores from hyperparameters if available
        f1_score_cv: Optional[float] = None
        f1_score_train: Optional[float] = None

        # Try to get scores from model-specific hyperparameters
        hyperparams = getattr(self, 'hyperparameters', None)

        if hyperparams is not None:
            f1_score_cv = getattr(hyperparams, 'f1_score_cv', None)
            f1_score_train = getattr(hyperparams, 'f1_score_train', None)

        return (ModelConfiguration(
            model_type=model_type,
            model_balancing=model_balancing,
            evaluation_method=ModelEvaluationMethod.Confusion_Matrix,
            params=params,
            data_splits=self.data_splits,
            cv=self.cv,
            class_weight=self.class_weight,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_iter=self.n_iter,
            features=self.data_splits.features_valid.columns.tolist(),
            f1_score_cv=f1_score_cv,
            f1_score_train=f1_score_train,
            f1_score_valid=f1_result,
            acceptable_gap=self.acceptable_gap,
            large_gap=self.large_gap
        ), report_text)
