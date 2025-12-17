from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional, cast, Mapping, Type, Any, Final
from dsr_feature_eng_ml.enums import ModelBalancing
from dsr_feature_eng_ml.evaluation.model_results import ModelResults
from dsr_feature_eng_ml.evaluation.data_splits import DataSplits
from dsr_feature_eng_ml.models.decision_tree import DecisionTree
from dsr_feature_eng_ml.models.random_forest import RandomForest
from dsr_feature_eng_ml.models.logistic_regression import LogisticRegression
from dsr_feature_eng_ml.constants import DEFAULT_LARGE_GAP, DEFAULT_ACCEPTABLE_GAP


class ModelEvaluation:
    """Orchestrates evaluation of multiple model types across different balancing strategies.

    Manages the complete model evaluation workflow, including training Decision Tree,
    Random Forest, and Logistic Regression models with various data balancing approaches
    (imbalanced, upsampled, downsampled). Tracks best performing models and determines
    which model types should be included in further evaluations based on F1 score gaps.

    Class Attributes:
        phase_number (int): Class-level counter tracking the current evaluation phase across
            all instances. Used to identify and organize sequential evaluation phases in the
            workflow (default: 0).

    Attributes:
        data_splits (DataSplits): Dataset splits for training, validation, and testing.
        dtree_param_grid (dict): Hyperparameter grid for Decision Tree models.
        rf_param_grid (dict): Hyperparameter grid for Random Forest models.
        lr_param_grid (dict): Hyperparameter grid for Logistic Regression models.
        cv (int): Number of cross-validation folds.
        n_iter (int): Number of iterations for randomized search.
        scoring (str): Scoring metric to optimize (e.g., 'f1').
        n_jobs (int): Number of parallel jobs for training (-1 uses all processors).
        viable_f1_gap (float): Maximum F1 score gap to keep model in evaluation.
        acceptable_gap (float): Minimum gap for acceptable generalization (default: DEFAULT_ACCEPTABLE_GAP).
        large_gap (float): Minimum gap indicating overfitting (default: DEFAULT_LARGE_GAP).        
        is_baseline_evaluation (bool): Whether this is the initial baseline evaluation.
        include_dtree (bool): Whether to include Decision Tree in evaluation.
        include_rf (bool): Whether to include Random Forest in evaluation.
        include_lr (bool): Whether to include Logistic Regression in evaluation.
        best_dtree_f1 (Optional[float]): Best Decision Tree F1 score so far.
        best_rf_f1 (Optional[float]): Best Random Forest F1 score so far.
        best_lr_f1 (Optional[float]): Best Logistic Regression F1 score so far.
        best_dtree_f1_overall (float): Overall best Decision Tree F1 across all evaluations.
        best_rf_f1_overall (float): Overall best Random Forest F1 across all evaluations.
        best_lr_f1_overall (float): Overall best Logistic Regression F1 across all evaluations.
        best_f1_overall (float): Overall best F1 score across all models and evaluations.

    Example:
        >>> evaluator = ModelEvaluation(
        ...     data_splits=splits,
        ...     dtree_param_grid={'max_depth': [None, 10, 20]},
        ...     rf_param_grid={'n_estimators': [10, 50, 100]},
        ...     lr_param_grid={'C': [0.1, 1.0, 10.0]},
        ...     cv=5,
        ...     n_iter=50,
        ...     scoring='f1',
        ...     n_jobs=-1,
        ...     viable_f1_gap=0.01
        ... )
        >>> evaluator.evaluate_models(
        ...     model_results=results,
        ...     step_description='Imbalanced',
        ...     model_balancing=ModelBalancing.Imbalance
        ... )
    """
    phase_number: int = 0

    def __init__(
            self,
            data_splits: DataSplits,
            dtree_param_grid: dict,
            rf_param_grid: dict,
            lr_param_grid: dict,
            cv: int,
            n_iter: int,
            scoring: str,
            n_jobs: int,
            viable_f1_gap: float,
            acceptable_gap: float = DEFAULT_ACCEPTABLE_GAP,
            large_gap: float = DEFAULT_LARGE_GAP,
            max_iter: int = 300,
            is_baseline_evaluation: bool = True,
            include_dtree: bool = True,
            include_rf: bool = True,
            include_lr: bool = True,
            best_dtree_f1: Optional[float] = None,
            best_rf_f1: Optional[float] = None,
            best_lr_f1: Optional[float] = None,
            best_dtree_f1_overall: float = 0.0,
            best_rf_f1_overall: float = 0.0,
            best_lr_f1_overall: float = 0.0,
            best_f1_overall: float = 0.0,
    ):
        self.data_splits = data_splits
        self.dtree_param_grid = dtree_param_grid
        self.rf_param_grid = rf_param_grid
        self.lr_param_grid = lr_param_grid
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.viable_f1_gap = viable_f1_gap
        self.acceptable_gap = acceptable_gap
        self.large_gap = large_gap
        self.max_iter = max_iter
        self.is_baseline_evaluation = is_baseline_evaluation
        self.include_dtree = include_dtree
        self.include_rf = include_rf
        self.include_lr = include_lr
        self.best_dtree_f1 = best_dtree_f1
        self.best_rf_f1 = best_rf_f1
        self.best_lr_f1 = best_lr_f1
        self.best_dtree_f1_overall = best_dtree_f1_overall
        self.best_rf_f1_overall = best_rf_f1_overall
        self.best_lr_f1_overall = best_lr_f1_overall
        self.best_f1_overall = best_f1_overall

    @classmethod
    def from_model_evaluator(
        cls,
        src: ModelEvaluation,
        data_splits: Optional[DataSplits] = None,
        dtree_param_grid: Optional[dict] = None,
        rf_param_grid: Optional[dict] = None,
        lr_param_grid: Optional[dict] = None,
        cv: Optional[int] = None,
        n_iter: Optional[int] = None,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
        viable_f1_gap: Optional[float] = None,
        acceptable_gap: Optional[float] = None,
        large_gap: Optional[float] = None,
        max_iter: Optional[int] = None,
        is_baseline_evaluation: Optional[bool] = None,
        include_dtree: Optional[bool] = None,
        include_rf: Optional[bool] = None,
        include_lr: Optional[bool] = None,
        best_dtree_f1_overall: Optional[float] = None,
        best_rf_f1_overall: Optional[float] = None,
        best_lr_f1_overall: Optional[float] = None,
        best_f1_overall: Optional[float] = None
    ) -> ModelEvaluation:
        """Create a new ModelEvaluation by copying and optionally modifying an existing one.

        Copies all settings from a source evaluator and allows selective override of
        specific parameters. Useful for creating variants with different data splits
        or hyperparameter grids while preserving other configuration.

        Args:
            src (ModelEvaluation): Source evaluator to copy from.
            data_splits (Optional[DataSplits]): Override data splits, or None to use src's.
            dtree_param_grid (Optional[dict]): Override DT param grid, or None to use src's.
            rf_param_grid (Optional[dict]): Override RF param grid, or None to use src's.
            lr_param_grid (Optional[dict]): Override LR param grid, or None to use src's.
            cv (Optional[int]): Override CV folds, or None to use src's.
            n_iter (Optional[int]): Override n_iter, or None to use src's.
            scoring (Optional[str]): Override scoring metric, or None to use src's.
            n_jobs (Optional[int]): Override n_jobs, or None to use src's.
            viable_f1_gap (Optional[float]): Override F1 gap, or None to use src's.
            acceptable_gap (Optional[float]): Override acceptable gap, or None to use src's.
            large_gap (Optional[float]): Override large gap, or None to use src's.
            max_iter (Optional[int]): Override max_iter, or None to use src's.            
            is_baseline_evaluation (Optional[bool]): Override baseline flag, or None for False.
            include_dtree (Optional[bool]): Override DT inclusion, or None to use src's.
            include_rf (Optional[bool]): Override RF inclusion, or None to use src's.
            include_lr (Optional[bool]): Override LR inclusion, or None to use src's.
            best_dtree_f1_overall (Optional[float]): Override best DT F1, or None to use src's.
            best_rf_f1_overall (Optional[float]): Override best RF F1, or None to use src's.
            best_lr_f1_overall (Optional[float]): Override best LR F1, or None to use src's.
            best_f1_overall (Optional[float]): Override best overall F1, or None to use src's.

        Returns:
            ModelEvaluation: New evaluator with specified overrides.
        """
        if data_splits is None:
            data_splits = src.data_splits

        if dtree_param_grid is None:
            dtree_param_grid = src.dtree_param_grid

        if rf_param_grid is None:
            rf_param_grid = src.rf_param_grid

        if lr_param_grid is None:
            lr_param_grid = src.lr_param_grid

        if cv is None:
            cv = src.cv

        if n_iter is None:
            n_iter = src.n_iter

        if scoring is None:
            scoring = src.scoring

        if n_jobs is None:
            n_jobs = src.n_jobs

        if viable_f1_gap is None:
            viable_f1_gap = src.viable_f1_gap

        if acceptable_gap is None:
            acceptable_gap = src.acceptable_gap

        if large_gap is None:
            large_gap = src.large_gap

        if max_iter is None:
            max_iter = src.max_iter

        if is_baseline_evaluation is None:
            is_baseline_evaluation = False

        if include_dtree is None:
            include_dtree = src.include_dtree

        if include_rf is None:
            include_rf = src.include_rf

        if include_lr is None:
            include_lr = src.include_lr

        if best_dtree_f1_overall is None:
            best_dtree_f1_overall = src.best_dtree_f1_overall

        if best_rf_f1_overall is None:
            best_rf_f1_overall = src.best_rf_f1_overall

        if best_lr_f1_overall is None:
            best_lr_f1_overall = src.best_lr_f1_overall

        if best_f1_overall is None:
            best_f1_overall = src.best_f1_overall

        new_me = ModelEvaluation(
            data_splits=data_splits,
            dtree_param_grid=dtree_param_grid,
            rf_param_grid=rf_param_grid,
            lr_param_grid=lr_param_grid,
            cv=cv,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            viable_f1_gap=viable_f1_gap,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap,
            max_iter=max_iter,
            is_baseline_evaluation=is_baseline_evaluation,
            include_dtree=include_dtree,
            include_rf=include_rf,
            include_lr=include_lr,
            best_dtree_f1_overall=best_dtree_f1_overall,
            best_rf_f1_overall=best_rf_f1_overall,
            best_lr_f1_overall=best_lr_f1_overall,
            best_f1_overall=best_f1_overall
        )

        return new_me

    @classmethod
    def evaluate_dataset(
        cls,
        data_splits: DataSplits,
        dtree_param_grid: dict,
        rf_param_grid: dict,
        lr_param_grid: dict,
        cv: int,
        n_iter: int,
        scoring: str,
        n_jobs: int,
        viable_f1_gap: float,
        report_title: str,
        perform_dtree_feature_selection: bool,
        perform_rf_feature_selection: bool,
        evaluate_decision_tree: bool = True,
        evaluate_random_forest: bool = True,
        evaluate_logistic_regression: bool = True,
        perform_imbalance: bool = True,
        perform_auto_balance: bool = True,
        perform_upsampling: bool = True,
        perform_downsampling: bool = True,
        acceptable_gap: float = DEFAULT_ACCEPTABLE_GAP,
        large_gap: float = DEFAULT_LARGE_GAP,
        max_iter: int = 300,
    ) -> ModelResults:
        """Comprehensive dataset evaluation across all balancing strategies.

        Orchestrates a complete model evaluation workflow by training and evaluating
        Decision Tree, Random Forest, and Logistic Regression models across four
        different class balancing strategies: Imbalance (original data), Auto Balanced
        (sklearn's balanced class weights), Upsampled (minority class oversampling),
        and Downsampled (majority class undersampling). Automatically tracks best
        performing models and determines viability for continued evaluation.

        Args:
            data_splits (DataSplits): Dataset splits for training, validation, and testing.
            dtree_param_grid (dict): Hyperparameter grid for Decision Tree models.
            rf_param_grid (dict): Hyperparameter grid for Random Forest models.
            lr_param_grid (dict): Hyperparameter grid for Logistic Regression models.
            cv (int): Number of cross-validation folds.
            n_iter (int): Number of iterations for randomized search.
            scoring (str): Scoring metric to optimize (e.g., 'f1', 'accuracy').
            n_jobs (int): Number of parallel jobs for training (-1 uses all processors).
            viable_f1_gap (float): Maximum F1 score gap to keep model in evaluation.
            acceptable_gap (float): Minimum gap for acceptable generalization (default: DEFAULT_ACCEPTABLE_GAP).
            large_gap (float): Minimum gap indicating overfitting (default: DEFAULT_LARGE_GAP).
            max_iter (int): Maximum iterations for iterative solvers like LogisticRegression (default: 300).
            report_title (str): The title to place at the top of the report.
            perform_dtree_feature_selection (bool): Whether calc_best_top_n should be called
                for Decision Tree models. See the DecisionTree.evaluate_model for further
                information on this parameter.
            perform_rf_feature_selection (bool): Whether calc_best_top_n should be called
                for Random Forest models. See the RandomForest.evaluate_model for further
                information on this parameter.
            evaluate_decision_tree (bool): Whether to include Decision Tree models in the evaluation.
            evaluate_random_forest (bool): Whether to include Random Forest models in the evaluation.
            evaluate_logistic_regression (bool): Whether to include Logistic Regression models in the evaluation.
            perform_imbalance (bool): Whether to include Imbalance in the evaluation.
            perform_auto_balance (bool): Whether to include Auto Balancing in the evaluation.
            perform_upsampling (bool): Whether to include Upsampling in the evaluation.
            perform_downsampling (bool): Whether to include Downsampling in the evaluation.

        Returns:
            ModelResults: Complete results object containing all model configurations,
                           best performers, and evaluation summaries across all strategies.

        Example:
            >>> results = ModelEvaluation.evaluate_dataset(
            ...     data_splits=splits,
            ...     dtree_param_grid={'max_depth': [10, 20]},
            ...     rf_param_grid={'n_estimators': [50, 100]},
            ...     lr_param_grid={'C': [0.1, 1.0]},
            ...     cv=5,
            ...     n_iter=10,
            ...     scoring='f1',
            ...     n_jobs=-1,
            ...     viable_f1_gap=0.05
            ... )
            >>> print(results.results_text)  # View comprehensive results
            >>> print(results.summary_text)  # View summary only
        """
        model_results = ModelResults()

        model_results.model_evaluation_title(
            report_title=report_title,
            border_char='*'
        )

        model_evaluator = ModelEvaluation(
            data_splits=data_splits,
            dtree_param_grid=dtree_param_grid,
            rf_param_grid=rf_param_grid,
            lr_param_grid=lr_param_grid,
            cv=cv,
            n_iter=n_iter,
            max_iter=max_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            viable_f1_gap=viable_f1_gap,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap
        )

        # Imbalance
        if perform_imbalance:
            model_evaluator.evaluate_models(
                model_results=model_results,
                step_description='Imbalance',
                model_balancing=ModelBalancing.Imbalance,
                class_weight=None,
                perform_dtree_feature_selection=perform_dtree_feature_selection,
                perform_rf_feature_selection=perform_rf_feature_selection,
                evaluate_decision_tree=evaluate_decision_tree,
                evaluate_random_forest=evaluate_random_forest,
                evaluate_logistic_regression=evaluate_logistic_regression
            )

        # Auto Balanced
        if perform_auto_balance:
            model_evaluator.evaluate_models(
                model_results=model_results,
                step_description='Auto Balanced',
                model_balancing=ModelBalancing.Auto_Balanced,
                class_weight='balanced',
                perform_dtree_feature_selection=perform_dtree_feature_selection,
                perform_rf_feature_selection=perform_rf_feature_selection,
                evaluate_decision_tree=evaluate_decision_tree,
                evaluate_random_forest=evaluate_random_forest,
                evaluate_logistic_regression=evaluate_logistic_regression
            )

        # Upsampling
        if perform_upsampling:
            upsampled_data_splits: DataSplits = DataSplits.from_data_splits(
                src=data_splits,
                features_to_include=data_splits.features_to_include
            )

            upsampled_data_splits.with_upsampled_training()

            upsampled_model_evaluator = ModelEvaluation.from_model_evaluator(
                src=model_evaluator,
                data_splits=upsampled_data_splits
            )

            upsampled_model_evaluator.evaluate_models(
                model_results=model_results,
                step_description='Upsampled',
                model_balancing=ModelBalancing.Upsampled,
                class_weight=None,
                perform_dtree_feature_selection=perform_dtree_feature_selection,
                perform_rf_feature_selection=perform_rf_feature_selection,
                evaluate_decision_tree=evaluate_decision_tree,
                evaluate_random_forest=evaluate_random_forest,
                evaluate_logistic_regression=evaluate_logistic_regression
            )

        # Downsampling
        if perform_downsampling:
            downsampled_data_splits: DataSplits = data_splits.with_downsampled_training()

            downsampled_model_evaluator = ModelEvaluation.from_model_evaluator(
                src=model_evaluator,
                data_splits=downsampled_data_splits
            )

            downsampled_model_evaluator.evaluate_models(
                model_results=model_results,
                step_description='Downsampled',
                model_balancing=ModelBalancing.Downsampled,
                class_weight=None,
                perform_dtree_feature_selection=perform_dtree_feature_selection,
                perform_rf_feature_selection=perform_rf_feature_selection,
                evaluate_decision_tree=evaluate_decision_tree,
                evaluate_random_forest=evaluate_random_forest,
                evaluate_logistic_regression=evaluate_logistic_regression
            )

        return model_results

    def reset_baseline(
            self
    ):
        """Reset evaluation state to initial baseline conditions.

        Re-enables all model types and resets overall F1 scores to zero, typically
        called when starting a new baseline evaluation after feature changes.
        """
        self.is_baseline_evaluation = True
        self.include_dtree = True
        self.include_rf = True
        self.include_lr = True
        self.best_dtree_f1_overall = 0.0
        self.best_rf_f1_overall = 0.0
        self.best_lr_f1_overall = 0.0
        self.best_f1_overall = 0.0

    def evaluate_models(
            self,
            model_results: ModelResults,
            step_description: str,
            model_balancing: ModelBalancing,
            class_weight: Optional[str],
            perform_dtree_feature_selection: bool,
            perform_rf_feature_selection: bool,
            evaluate_decision_tree: bool = True,
            evaluate_random_forest: bool = True,
            evaluate_logistic_regression: bool = True,
    ):
        """Evaluate all enabled model types and track best performing models.

        Orchestrates training and evaluation of Decision Tree, Random Forest, and
        Logistic Regression models using the specified data balancing strategy.
        Updates best F1 scores and determines which models remain viable based on
        the F1 gap threshold.

        Args:
            model_results (ModelResults): Results accumulator for all model configurations.
            step_description (str): Descriptive name for this evaluation step.
            model_balancing (ModelBalancing): Data balancing strategy to apply.
            class_weight (Optional[str]): Class weight setting ('balanced' or None).
            perform_dtree_feature_selection (bool): Whether calc_best_top_n should be called
                for Decision Tree models. See the DecisionTree.evaluate_model for further
                information on this parameter.
            perform_rf_feature_selection (bool): Whether calc_best_top_n should be called
                for Random Forest models. See the RandomForest.evaluate_model for further
                information on this parameter.
            evaluate_decision_tree (bool): Whether to include Decision Tree models in the evaluation.
            evaluate_random_forest (bool): Whether to include Random Forest models in the evaluation.
            evaluate_logistic_regression (bool): Whether to include Logistic Regression models in the evaluation.

        Note:
            Only models with include_dtree/include_rf/include_lr=True are evaluated.
            Best F1 scores are tracked per model type and overall.
        """
        model_results.model_evaluation_header(
            step_description=step_description)

        if self.is_baseline_evaluation:
            self.reset_baseline()
            self.is_baseline_evaluation = False

        self.best_dtree_f1 = None
        self.best_rf_f1 = None
        self.best_lr_f1 = None

        # Decision Tree
        if evaluate_decision_tree and self.include_dtree:
            dtree = DecisionTree(
                data_splits=self.data_splits,
                cv=self.cv,
                param_grid=self.dtree_param_grid,
                class_weight=class_weight,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                n_iter=self.n_iter,
                acceptable_gap=self.acceptable_gap,
                large_gap=self.large_gap,
            )

            mc_list, report_text = dtree.evaluate_model(
                model_balancing=model_balancing,
                perform_feature_selection=perform_dtree_feature_selection,
                is_first_model=True
            )

            model_results.process_results(
                results=mc_list,
                report_text=report_text,
                step_title=f'Decision Tree ({step_description})'
            )

            best_f1_score = model_results.best_model_configuration_last_results.f1_score_valid

            if self.best_dtree_f1 is None or best_f1_score > self.best_dtree_f1:
                self.best_dtree_f1 = best_f1_score

            if best_f1_score > self.best_dtree_f1_overall:
                self.best_dtree_f1_overall = best_f1_score

            if self.best_dtree_f1_overall > self.best_f1_overall:
                self.best_f1_overall = self.best_dtree_f1_overall

        # Random Forest
        if evaluate_random_forest and self.include_rf:
            rf = RandomForest(
                data_splits=self.data_splits,
                cv=self.cv,
                n_iter=self.n_iter,
                param_grid=self.rf_param_grid,
                class_weight=class_weight,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                acceptable_gap=self.acceptable_gap,
                large_gap=self.large_gap,
            )

            mc_list, report_text = rf.evaluate_model(
                model_balancing=model_balancing,
                perform_feature_selection=perform_rf_feature_selection,
                is_first_model=False
            )

            model_results.process_results(
                results=mc_list,
                report_text=report_text,
                step_title=f'Random Forest ({step_description})'
            )

            best_f1_score = model_results.best_model_configuration_last_results.f1_score_valid

            if self.best_rf_f1 is None or best_f1_score > self.best_rf_f1:
                self.best_rf_f1 = best_f1_score

            if best_f1_score > self.best_rf_f1_overall:
                self.best_rf_f1_overall = best_f1_score

            if self.best_rf_f1_overall > self.best_f1_overall:
                self.best_f1_overall = self.best_rf_f1_overall

        # Logistic Regression
        if evaluate_logistic_regression and self.include_lr:
            lr = LogisticRegression(
                data_splits=self.data_splits,
                cv=self.cv,
                param_grid=self.lr_param_grid,
                class_weight=class_weight,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                n_iter=self.n_iter,
                max_iter=self.max_iter,
                acceptable_gap=self.acceptable_gap,
                large_gap=self.large_gap,
            )

            mc_list, report_text = lr.evaluate_model(
                model_balancing=model_balancing,
                is_first_model=False
            )

            model_results.process_results(
                results=mc_list,
                report_text=report_text,
                step_title=f'Logistic Regression ({step_description})'
            )

            best_f1_score = model_results.best_model_configuration_last_results.f1_score_valid

            if self.best_lr_f1 is None or best_f1_score > self.best_lr_f1:
                self.best_lr_f1 = best_f1_score

            if best_f1_score > self.best_lr_f1_overall:
                self.best_lr_f1_overall = best_f1_score

            if self.best_lr_f1_overall > self.best_f1_overall:
                self.best_f1_overall = self.best_lr_f1_overall

        best_f1_score = 0.0

        if self.best_dtree_f1 is not None:
            best_f1_score = self.best_dtree_f1

        if self.best_rf_f1 is not None and self.best_rf_f1 > best_f1_score:
            best_f1_score = self.best_rf_f1

        if self.best_lr_f1 is not None and self.best_lr_f1 > best_f1_score:
            best_f1_score = self.best_lr_f1

        if self.include_dtree and self.best_dtree_f1 is not None:
            self.include_dtree = (
                (self.best_f1_overall - self.best_dtree_f1_overall) < self.viable_f1_gap)
        else:
            self.include_dtree = False

        if self.include_rf and self.best_rf_f1 is not None:
            self.include_rf = (
                (self.best_f1_overall - self.best_rf_f1_overall) < self.viable_f1_gap)
        else:
            self.include_rf = False

        if self.include_lr and self.best_lr_f1 is not None:
            self.include_lr = (
                (self.best_f1_overall - self.best_lr_f1_overall) < self.viable_f1_gap)
        else:
            self.include_lr = False

        model_results.model_evaluation_footer(
            step_description=step_description,
            best_f1=best_f1_score,
            best_f1_overall=self.best_f1_overall,
            best_dtree_f1=self.best_dtree_f1,
            best_rf_f1=self.best_rf_f1,
            best_lr_f1=self.best_lr_f1,
            best_dtree_f1_overall=self.best_dtree_f1_overall,
            best_rf_f1_overall=self.best_rf_f1_overall,
            best_lr_f1_overall=self.best_lr_f1_overall,
            include_dtree=self.include_dtree,
            include_rf=self.include_rf,
            include_lr=self.include_lr,
            viable_f1_gap=self.viable_f1_gap,
        )
