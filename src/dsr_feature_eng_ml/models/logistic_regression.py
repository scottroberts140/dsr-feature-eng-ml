from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Union, Optional, cast, Mapping, Type, Any, Final
from dsr_feature_eng_ml.enums import ModelGeneralization, ModelType, ModelBalancing, ModelEvaluationMethod
from dsr_feature_eng_ml.models.model_specification import ModelSpecification
from dsr_feature_eng_ml.evaluation.data_splits import DataSplits
from dsr_feature_eng_ml.evaluation.feature_importance import ModelFeatureImportance
from dsr_feature_eng_ml.evaluation.model_configuration import ModelConfiguration
from dsr_feature_eng_ml.evaluation.validation_results import ValidationResults
from dsr_feature_eng_ml.constants import DEFAULT_LARGE_GAP, DEFAULT_ACCEPTABLE_GAP, REPORT_WIDTH, F1_FORMAT
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve


@dataclass(frozen=True)
class LogisticRegressionHyperParameters:
    """Immutable container for Logistic Regression hyperparameters and performance metrics.

    Attributes:
        C (float): Inverse of regularization strength (smaller = stronger).
        solver (str): Optimization algorithm used.
        l1_ratio (Optional[float]): Ratio of L1 regularization (0.0=L2, 1.0=L1, None=default).
        f1_score_cv (Optional[float]): F1 score from cross-validation (hyperparameter search).
        f1_score_train (Optional[float]): F1 score on training set.
        f1_score_valid (float): F1 score on validation set (primary comparison metric).
        model_generalization (ModelGeneralization): Generalization quality, calculated
            automatically based on the gap between training and validation F1 scores.

    Example:
        >>> hyperparams = LogisticRegressionHyperParameters(
        ...     C=1.0,
        ...     solver='lbfgs',
        ...     l1_ratio=0.0,
        ...     f1_score_cv=0.82,
        ...     f1_score_train=0.82,
        ...     f1_score_valid=0.82,
        ... )

    Note:
        This is an immutable dataclass (frozen=True). Once created, attributes
        cannot be modified. Use the from_search_cv() factory method to create
        instances from RandomizedSearchCV results.
    """
    C: float
    solver: str
    l1_ratio: Optional[float]
    f1_score_cv: Optional[float]
    f1_score_train: Optional[float]
    f1_score_valid: Optional[float]
    model_generalization: ModelGeneralization = ModelGeneralization.Undefined

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

    @property
    def params(self):
        return {
            'C': self.C,
            'l1_ratio': self.l1_ratio,
            'solver': self.solver
        }

    @classmethod
    def empty(
        cls
    ) -> LogisticRegressionHyperParameters:
        """Create an empty placeholder instance.

        Returns a sentinel instance with default values, typically used for
        initialization before actual hyperparameters are computed.

        Returns:
            LogisticRegressionHyperParameters: Empty instance with C=0.0, solver='',
            l1_ratio=None, f1_score=None.
        """
        return cls(
            C=0.0,
            solver='',
            l1_ratio=None,
            f1_score_cv=None,
            f1_score_train=None,
            f1_score_valid=None
        )

    @classmethod
    def from_search_cv(
        cls,
        search_cv: Union[GridSearchCV, RandomizedSearchCV]
    ) -> LogisticRegressionHyperParameters:
        """Extract hyperparameters from a completed GridSearchCV or RandomizedSearchCV object.

        Args:
            search_cv (GridSearchCV or RandomizedSearchCV): Fitted search object.

        Returns:
            LogisticRegressionHyperParameters: Hyperparameters from best estimator.
        """
        C = search_cv.best_params_['C']
        solver = search_cv.best_params_['solver']
        l1_ratio = search_cv.best_params_.get('l1_ratio', None)
        f1_score_cv = search_cv.best_score_

        return cls(
            C=C,
            solver=solver,
            l1_ratio=l1_ratio,
            f1_score_cv=f1_score_cv,
            f1_score_train=None,
            f1_score_valid=None
        )

    def info(
            self,
            header: str
    ) -> str:
        """Display hyperparameter information with a custom header.

        Args:
            header (str): Title to display above the hyperparameter info.
        """
        return f'''{header.center(REPORT_WIDTH, '-')}
Parameters: {self.params}
CV F1 score:             {self.f1_score_cv:{F1_FORMAT}}
Training Set F1 score:   {self.f1_score_train:{F1_FORMAT}}
Validation Set F1 score: {self.f1_score_valid:{F1_FORMAT}}
Model Generalization:    {self.model_generalization.name}'''


class LogisticRegression(ModelSpecification):
    """Logistic Regression model training and evaluation pipeline.

    Handles hyperparameter tuning with GridSearchCV, model validation, and test
    set evaluation for logistic regression classifiers. Note: Does not perform
    feature importance analysis like tree-based models.

    This class uses managed mutability: attributes are initialized with sentinel
    values and updated through workflow methods (tune_hyperparameters, validate_model, etc.).

    Attributes:
        param_grid (dict): Grid of hyperparameters for search.
        hyperparameters (LogisticRegressionHyperParameters): Best params.
        validation_results (ValidationResults): Validation predictions (initialized as empty).
        test_model (LogisticRegression): Model for test set evaluation (initialized as sentinel).
        acceptable_gap (float): Minimum gap for acceptable generalization (default: DEFAULT_ACCEPTABLE_GAP).
        large_gap (float): Minimum gap indicating overfitting (default: DEFAULT_LARGE_GAP).

    Example:
        >>> lr = LogisticRegression(
        ...     data_splits=splits,
        ...     cv=5,
        ...     param_grid={'C': [0.1, 1.0, 10.0],
        ...                 'max_iter': [100, 200]},
        ...     class_weight='balanced'
        ... )
        >>> lr.tune_hyperparameters()  # Workflow methods update attributes
        >>> lr.validate_model(ModelBalancing.Auto_Balanced)

    Note:
        Attributes are initialized with sentinel values and are updated by calling
        workflow methods in sequence: tune_hyperparameters() → validate_model() → etc.
        Unlike tree-based models, logistic regression does not compute feature importance.
    """

    def __init__(
            self,
            data_splits: DataSplits,
            cv: int,
            param_grid: dict,
            class_weight: Optional[Union[
                Mapping[str, float],
                str
            ]] = None,
            scoring: str = 'f1',
            n_iter: int = 0,
            n_jobs: int = -1,
            max_iter: int = 300,
            acceptable_gap: float = DEFAULT_ACCEPTABLE_GAP,
            large_gap: float = DEFAULT_LARGE_GAP,
    ):
        super().__init__(
            data_splits=data_splits,
            cv=cv,
            class_weight=class_weight,
            scoring=scoring,
            n_jobs=n_jobs,
            n_iter=n_iter,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap
        )

        self.param_grid = param_grid
        self.max_iter = max_iter
        self.hyperparameters = LogisticRegressionHyperParameters.empty()
        self.validation_results = ValidationResults.empty()
        self.validation_model_feature_importances = ModelFeatureImportance.empty(
            ModelType.Logistic_Regression)
        self.test_model = SklearnLogisticRegression()

    @classmethod
    def model_from_hyperparameters(
        cls,
        params: dict,
        random_state: int,
        class_weight: Optional[Union[
            Mapping[str, float],
            str
        ]],
        max_iter: int = 300,
    ) -> SklearnLogisticRegression:
        """Create a LogisticRegression model from hyperparameters dictionary.

        Factory method that constructs a configured LogisticRegression instance
        using hyperparameters from a dictionary. The returned model is configured
        but not yet trained.

        Args:
            params (dict): Hyperparameters with keys 'C', 'penalty', and 'solver'.
            random_state (int): Random seed for reproducibility.
            class_weight (Optional[Union[Mapping[str, float], str]]): Class balancing strategy.
                Can be 'balanced', a dict mapping class labels to weights, or None.

        Returns:
            SklearnLogisticRegression: Configured but untrained logistic regression model.

        Example:
            >>> params = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
            >>> model = LogisticRegression.model_from_hyperparameters(params, random_state=42, class_weight='balanced')
            >>> model.fit(X_train, y_train)
        """

        return SklearnLogisticRegression(
            random_state=random_state,
            C=params['C'],
            solver=params['solver'],  # pyright: ignore[reportArgumentType]
            class_weight=class_weight,  # pyright: ignore[reportArgumentType]
            l1_ratio=params.get('l1_ratio'),
            max_iter=max_iter
        )

    @classmethod
    def optimal_threshold(
        cls,
        data_splits: DataSplits,
        model: SklearnLogisticRegression,
    ) -> tuple[float, float, np.ndarray]:
        """Find optimal classification threshold by maximizing F1 score on validation set.

        Trains the model, generates probability predictions on the validation set,
        and finds the threshold that maximizes F1 score using precision-recall curve.
        Then evaluates the model on the test set using this optimal threshold.

        Args:
            data_splits (DataSplits): Training, validation, and test data splits.
            model (LogisticRegression): Untrained logistic regression to optimize.

        Returns:
            tuple[float, float, np.ndarray]: A tuple containing:
                - optimal_threshold: The threshold value that maximizes validation F1 score
                - final_f1_result: The F1 score achieved on test set using optimal threshold
                - test_proba: Array of predicted probabilities for positive class on test set

        Note:
            This method fits the model on training data, optimizes threshold on validation
            data, and evaluates on test data. The threshold is chosen to maximize F1 score,
            which may differ from the default 0.5 threshold, especially with imbalanced data.
        """
        model.fit(data_splits.features_train, data_splits.target_train)
        valid_proba = model.predict_proba(data_splits.features_valid)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(
            data_splits.target_valid,
            valid_proba
        )

        f1_scores = np.nan_to_num(
            2 * (precisions * recalls) / (precisions + recalls))

        optimal_threshold_index = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_threshold_index]

        test_proba = model.predict_proba(data_splits.features_test)[:, 1]
        y_pred_test = (test_proba >= optimal_threshold).astype(int)
        final_f1_result = cast(float, f1_score(
            data_splits.target_test, y_pred_test))

        return optimal_threshold, final_f1_result, test_proba

    def tune_hyperparameters(
            self,
    ) -> None:
        """Perform hyperparameter search to find optimal configuration.

        Automatically selects between RandomizedSearchCV (for distributions) or
        GridSearchCV (for discrete lists) based on param_grid structure.

        Trains multiple logistic regression models with different hyperparameter
        combinations using cross-validation on the training set.

        Note:
            Updates hyperparameters attribute.
            Uses RandomizedSearchCV if param_grid contains scipy distributions, otherwise
            uses GridSearchCV for discrete parameter lists.
        """
        search_cv_model = SklearnLogisticRegression(
            random_state=self.random_state,
            class_weight=self.class_weight,
            max_iter=self.max_iter
        )

        # Automatically detect whether to use RandomizedSearchCV or GridSearchCV
        use_randomized = any(
            hasattr(value, 'rvs')  # scipy distributions have .rvs() method
            for value in self.param_grid.values()
        )

        if use_randomized:
            # Use RandomizedSearchCV for distributions
            search_cv = RandomizedSearchCV(
                search_cv_model,
                self.param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )
        else:
            # Use GridSearchCV for discrete lists
            search_cv = GridSearchCV(
                search_cv_model,
                self.param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs
            )

        search_cv.fit(self.data_splits.features_train,
                      self.data_splits.target_train)
        self.hyperparameters = LogisticRegressionHyperParameters.from_search_cv(
            search_cv)

    def validate_model(
            self,
            model_balancing: ModelBalancing,
            header: str,
            include_in_report: bool = True,
    ) -> tuple[ModelConfiguration, str]:
        """Train model with best hyperparameters and evaluate on validation set.

        Uses the optimal hyperparameters from RandomizedSearchCV to train a new model
        and evaluates its performance on the training and validation sets.

        Args:
            model_balancing (ModelBalancing): The balancing strategy applied to data.
            header (str): Display header for report.
            include_in_report (bool): Whether to print validation results.

        Returns:
            ModelConfiguration: Configuration object with validation metrics.

        Note:
            Updates validation_results, hyperparameters (with train/valid F1 scores).
        """
        report_text = ''

        validation_model = SklearnLogisticRegression(
            random_state=self.random_state,
            C=self.hyperparameters.C,
            solver=self.hyperparameters.solver,  # type: ignore[arg-type]
            class_weight=self.class_weight,
            l1_ratio=self.hyperparameters.l1_ratio,
            max_iter=self.max_iter
        )

        validation_model.fit(self.data_splits.features_train,
                             self.data_splits.target_train)
        predicted_train = validation_model.predict(
            self.data_splits.features_train)
        f1_train_result: float = cast(float, f1_score(
            self.data_splits.target_train, predicted_train))
        predicted_valid = validation_model.predict(
            self.data_splits.features_valid)
        f1_valid_result: float = cast(float, f1_score(
            self.data_splits.target_valid, predicted_valid))

        self.hyperparameters = LogisticRegressionHyperParameters(
            C=self.hyperparameters.C,
            l1_ratio=self.hyperparameters.l1_ratio,
            solver=self.hyperparameters.solver,
            f1_score_cv=self.hyperparameters.f1_score_cv,
            f1_score_train=f1_train_result,
            f1_score_valid=f1_valid_result
        )

        self.validation_results = ValidationResults(
            predicted_valid=pd.Series(
                predicted_valid, index=self.data_splits.target_valid.index),
            predicted_train=pd.Series(
                predicted_train, index=self.data_splits.target_train.index)
        )

        if include_in_report:
            train_header = f'{header} (Training Set)'
            valid_header = f'{header} (Validation Set)'
            report_text += f'''
{train_header}
F1 score: {f1_train_result:{F1_FORMAT}}
'''

            report_text += self.hyperparameters.info(valid_header)

        optimized_params = {
            'C': self.hyperparameters.C,
            'l1_ratio': self.hyperparameters.l1_ratio,
            'solver': self.hyperparameters.solver
        }

        return (ModelConfiguration(
            model_type=ModelType.Logistic_Regression,
            model_balancing=model_balancing,
            evaluation_method=ModelEvaluationMethod.Randomized_Search,
            params=optimized_params,
            data_splits=self.data_splits,
            cv=self.cv,
            class_weight=self.class_weight,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_iter=self.n_iter,
            features=self.data_splits.features_valid.columns.tolist(),
            f1_score_cv=self.hyperparameters.f1_score_cv,
            f1_score_train=self.hyperparameters.f1_score_train,
            f1_score_valid=f1_valid_result,
            acceptable_gap=self.acceptable_gap,
            large_gap=self.large_gap
        ), report_text)

    def evaluate_model(
            self,
            model_balancing: ModelBalancing,
            is_first_model: bool = False,
            include_in_report: bool = True,
    ) -> tuple[list[ModelConfiguration], str]:
        """Complete model evaluation pipeline: search and validate.

        Executes the workflow: RandomizedSearchCV for hyperparameters and validation
        with best params. Note: No feature selection for logistic regression.

        Args:
            model_balancing (ModelBalancing): The balancing strategy for data.
            is_first_model (bool): Whether the model is the first to be evaluated.
                This parameter controls formatting for the report text.
            include_in_report (bool): Whether to print detailed results at each step.

        Returns:
            list[ModelConfiguration]: All model configurations tested.
        """
        report_text = ''

        if not is_first_model:
            report_text += '\n'

        mc_list: list[ModelConfiguration] = []
        self.tune_hyperparameters()

        mc, t = self.validate_model(
            model_balancing=model_balancing,
            include_in_report=include_in_report,
            header=f'Logistic Regression ({model_balancing.name} - All Features)'
        )

        mc_list.append(mc)
        report_text += t

        return mc_list, report_text

    def calc_confusion_matrix(
            self,
            model_balancing: ModelBalancing,
            include_in_report: bool = True,
            show_plot: bool = True
    ) -> tuple[list[ModelConfiguration], str]:
        """Calculate confusion matrix metrics for validation predictions.

        Generates predictions on validation set and computes detailed confusion
        matrix statistics including precision, recall, and F1 score.

        Args:
            model_balancing (ModelBalancing): The balancing strategy for data.
            include_in_report (bool): Whether to print confusion matrix and metrics.
            show_plot (bool): Whether to display target frequency plot.

        Returns:
            list[ModelConfiguration]: Configuration with confusion matrix results.
        """
        report_text = ''
        mc_list: list[ModelConfiguration] = []

        # Use stored predictions from validation_results
        self.predicted_valid = self.validation_results.predicted_valid
        target_frequency = self.predicted_valid.value_counts(normalize=True)

        if include_in_report:
            report_text += f'\n\nTarget Frequency: {target_frequency}\n'

        if show_plot:
            target_frequency.plot(
                kind='bar',
                y=self.data_splits.target_column,
                ylabel='Frequency',
                title=f'Target Frequency [{self.data_splits.target_column}] (Validation Set)'
            )

        mc, t = self.calc_target_frequency(
            model_type=ModelType.Logistic_Regression,
            model_balancing=model_balancing,
            params=self.param_grid,
            df=self.data_splits.target_valid,
            include_in_report=include_in_report
        )

        mc_list.append(mc)
        report_text += t

        return mc_list, t

    def prepare_test_model(
            self,
            model_configuration: ModelConfiguration,
            include_in_report: bool = True,
            header: str = 'Logistic Regression (Validation Set)'
    ) -> str:
        """Train final model on combined train+validation data for test evaluation.

        Args:
            include_in_report (bool): Whether to print test set hyperparameters.
            header (str): Display header for report.

        Note:
            Trains on combined train+valid sets to maximize training data before test.
        """
        report_text = ''

        self.test_model = SklearnLogisticRegression(
            random_state=self.random_state,
            C=self.param_grid['C'],
            solver=self.param_grid['solver'],
            l1_ratio=self.param_grid.get('l1_ratio'),
            max_iter=self.max_iter
        )

        self.test_model.fit(self.data_splits.features_train,
                            self.data_splits.target_train)

        if include_in_report:
            report_text += f'''{header.center(REPORT_WIDTH, '-')}
        {model_configuration.info()}
        '''

        return report_text

    @classmethod
    def evaluate_test_set(
            cls,
            model_configuration: ModelConfiguration
    ) -> tuple[float, str]:
        """Evaluate a model configuration on the held-out test set.

        Creates a new model with the given configuration's hyperparameters,
        trains on train+validation data, and evaluates on test set.

        Args:
            model_configuration (ModelConfiguration): Configuration to evaluate.

        Returns:
            float: The F1 score.
            str: Formatted test results including F1 score.

        Note:
            Prints F1 score.
        """
        text_result = ''

        lr = LogisticRegression(
            data_splits=model_configuration.data_splits,
            cv=model_configuration.cv,
            param_grid=model_configuration.params,
            class_weight=model_configuration.class_weight,
            scoring=model_configuration.scoring,
            n_jobs=model_configuration.n_jobs
        )

        text_result += lr.prepare_test_model(
            model_configuration=model_configuration,
            include_in_report=True,
            header='Best Model (Validation Set)'
        )

        predicted_test = lr.test_model.predict(
            model_configuration.data_splits.features_test)
        f1_result = cast(float, f1_score(
            model_configuration.data_splits.target_test, predicted_test))
        text_result += f'Test set F1 score: {f1_result:{F1_FORMAT}}'
        return f1_result, text_result
