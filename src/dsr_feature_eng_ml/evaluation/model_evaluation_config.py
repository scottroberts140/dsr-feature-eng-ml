"""
Model Evaluation Configuration

Provides a configuration class for encapsulating model evaluation parameters,
reducing code duplication and improving maintainability in analysis workflows.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class ModelEvaluationConfig:
    """Configuration container for model evaluation runs.

    Encapsulates all parameters for a single model evaluation, including data splits,
    model hyperparameter grids, evaluation settings, and reporting options. Supports
    auto-incrementing phase numbers for sequential evaluation workflows.

    Attributes:
        data_splits: Dataset splits for training, validation, and testing.
        dataset_name: Human-readable name for this dataset/phase (e.g., "Base Feature Set").
        dtree_param_grid: Hyperparameter grid for Decision Tree models.
        rf_param_grid: Hyperparameter grid for Random Forest models.
        lr_param_grid: Hyperparameter grid for Logistic Regression models.
        cv: Number of cross-validation folds.
        n_iter: Number of iterations for randomized search.
        max_iter: Maximum iterations for iterative solvers.
        scoring: Scoring metric to optimize (e.g., 'f1', 'accuracy').
        n_jobs: Number of parallel jobs (-1 uses all processors).
        viable_f1_gap: Maximum F1 score gap to keep model in evaluation.
        perform_dtree_feature_selection: Whether to perform feature selection for Decision Tree.
        perform_rf_feature_selection: Whether to perform feature selection for Random Forest.
        evaluate_decision_tree: Whether to evaluate Decision Tree models.
        evaluate_random_forest: Whether to evaluate Random Forest models.
        evaluate_logistic_regression: Whether to evaluate Logistic Regression models.
        perform_imbalance: Whether to evaluate with imbalanced data.
        perform_auto_balance: Whether to evaluate with auto-balanced class weights.
        perform_upsampling: Whether to evaluate with minority class upsampling.
        perform_downsampling: Whether to evaluate with majority class downsampling.
        acceptable_gap: Minimum gap for acceptable generalization.
        large_gap: Minimum gap indicating overfitting.
        phase_number: Explicit phase number (None to auto-increment).
        auto_increment_phase: Whether to auto-increment phase number.
    """

    # Core data
    data_splits: object  # dfem.DataSplits
    dataset_name: str

    # Model hyperparameter grids
    dtree_param_grid: dict
    rf_param_grid: dict
    lr_param_grid: dict

    # Evaluation parameters
    cv: int
    n_iter: int
    max_iter: int
    scoring: str
    n_jobs: int
    viable_f1_gap: float

    # Feature selection
    perform_dtree_feature_selection: bool = True
    perform_rf_feature_selection: bool = True

    # Model evaluation options
    evaluate_decision_tree: bool = True
    evaluate_random_forest: bool = True
    evaluate_logistic_regression: bool = True

    # Class balancing strategies
    perform_imbalance: bool = True
    perform_auto_balance: bool = True
    perform_upsampling: bool = True
    perform_downsampling: bool = True

    # Gap thresholds
    acceptable_gap: float = 0.05
    large_gap: float = 0.10

    # Phase numbering
    phase_number: Optional[int] = None
    auto_increment_phase: bool = True

    def __post_init__(self):
        """Handle phase number auto-increment if enabled."""
        # Avoid circular import by importing here
        import dsr_feature_eng_ml as dfem

        if self.auto_increment_phase:
            dfem.ModelEvaluation.phase_number += 1
            self.phase_number = dfem.ModelEvaluation.phase_number
        elif self.phase_number is None:
            self.phase_number = dfem.ModelEvaluation.phase_number

    @property
    def report_title(self) -> str:
        """Generate report title from phase number and dataset name.

        Returns:
            str: Formatted report title like "Phase 1 - Base Feature Set"
        """
        return f'Phase {self.phase_number} - {self.dataset_name}'

    @classmethod
    def from_dataset(
        cls,
        dataset: pd.DataFrame,
        target_column: str,
        dataset_name: str,
        config_params: dict,
        param_grids: dict,
        auto_increment: bool = True,
        **kwargs
    ) -> ModelEvaluationConfig:
        """Create evaluation config from a DataFrame.

        Generic factory method for creating configs from any dataset.

        Args:
            dataset: DataFrame to analyze.
            target_column: Name of the target column.
            dataset_name: Human-readable name for this dataset phase.
            config_params: Dict with keys: test_size, valid_size, random_state, cv, 
                          n_iter, max_iter, scoring, n_jobs, viable_f1_gap
            param_grids: Dict with keys: dtree, rf, lr (each containing hyperparameter grids)
            auto_increment: Whether to auto-increment phase number.
            **kwargs: Additional parameters to pass to evaluate_dataset (e.g., 
                     evaluate_decision_tree, perform_auto_balance, etc.)

        Returns:
            ModelEvaluationConfig: Configuration object ready for evaluation.

        Example:
            >>> config = ModelEvaluationConfig.from_dataset(
            ...     dataset=df_processed,
            ...     target_column='Exited',
            ...     dataset_name='Custom Features',
            ...     config_params=config_params,
            ...     param_grids=param_grids
            ... )
        """
        # Import here to avoid circular dependency
        import dsr_feature_eng_ml as dfem

        all_features = [col for col in dataset.columns if col != target_column]
        data_splits = dfem.DataSplits.from_data_source(
            src=dataset,
            features_to_include=all_features,
            target_column=target_column,
            test_size=config_params['test_size'],
            valid_size=config_params['valid_size'],
            random_state=config_params['random_state']
        )

        return cls(
            data_splits=data_splits,
            dataset_name=dataset_name,
            dtree_param_grid=param_grids['dtree'],
            rf_param_grid=param_grids['rf'],
            lr_param_grid=param_grids['lr'],
            cv=config_params['cv'],
            n_iter=config_params['n_iter'],
            max_iter=config_params['max_iter'],
            scoring=config_params['scoring'],
            n_jobs=config_params['n_jobs'],
            viable_f1_gap=config_params['viable_f1_gap'],
            auto_increment_phase=auto_increment,
            **kwargs
        )

