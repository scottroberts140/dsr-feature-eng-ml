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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_recall_curve


@dataclass(frozen=True)
class DecisionTreeHyperParameters:
    """Immutable container for Decision Tree hyperparameters and performance metrics.

    Attributes:
        max_depth (int): Maximum depth of the decision tree.
        min_samples_leaf (Union[int, float]): Minimum samples required at leaf node.
        min_samples_split (Union[int, float]): Minimum samples required to split.
        f1_score_cv (Optional[float]): F1 score from cross-validation (hyperparameter search).
        f1_score_train (Optional[float]): F1 score on training set.
        f1_score_valid (float): F1 score on validation set (primary comparison metric).
        model_generalization (ModelGeneralization): Generalization quality, calculated
            automatically based on the gap between training and validation F1 scores.

    Example:
        >>> hyperparams = DecisionTreeHyperParameters(
        ...     max_depth=10,
        ...     min_samples_leaf=5,
        ...     min_samples_split=10,
        ...     f1_score_cv=0.82,
        ...     f1_score_train=0.82,
        ...     f1_score_valid=0.82,
        ... )

    Note:
        This is an immutable dataclass (frozen=True). Once created, attributes
        cannot be modified. Use the from_search_cv() factory method to create
        instances from GridSearchCV results.
    """
    max_depth: int
    min_samples_leaf: Union[int, float]
    min_samples_split: Union[int, float]
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
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'min_samples_split': self.min_samples_split
        }

    @classmethod
    def empty(
        cls
    ) -> DecisionTreeHyperParameters:
        """Create an empty placeholder instance.

        Returns a sentinel instance with default values, typically used for
        initialization before actual hyperparameters are computed.

        Returns:
            DecisionTreeHyperParameters: Empty instance with max_depth=0,
            min_samples_leaf=0, min_samples_split=0, f1_score=None.
        """
        return cls(
            max_depth=0,
            min_samples_leaf=0,
            min_samples_split=0,
            f1_score_cv=None,
            f1_score_train=None,
            f1_score_valid=None
        )

    @classmethod
    def from_search_cv(
        cls,
        search_cv: Union[GridSearchCV, RandomizedSearchCV]
    ) -> DecisionTreeHyperParameters:
        """Extract hyperparameters from a completed GridSearchCV or RandomizedSearchCV object.

        Args:
            search_cv: Fitted GridSearchCV or RandomizedSearchCV object.

        Returns:
            DecisionTreeHyperParameters: Hyperparameters from best estimator.
        """
        max_depth = search_cv.best_params_['max_depth']
        min_samples_leaf = search_cv.best_params_['min_samples_leaf']
        min_samples_split = search_cv.best_params_['min_samples_split']
        f1_score_cv = search_cv.best_score_

        return cls(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
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


class DecisionTree(ModelSpecification):
    """Decision Tree model training and evaluation pipeline.

    Handles hyperparameter tuning with GridSearchCV, model validation, feature
    importance analysis, and test set evaluation for decision tree classifiers.

    This class uses managed mutability: attributes are initialized with sentinel
    values and updated through workflow methods (tune_hyperparameters, validate_model, etc.).

    Attributes:
        param_grid (dict): Grid of hyperparameters for search.
        hyperparameters (DecisionTreeHyperParameters): Best params from CV (initialized as empty).
        validation_results (ValidationResults): Validation predictions (initialized as empty).
        validation_model_feature_importances (ModelFeatureImportance): Feature importance (initialized as empty).
        test_model (DecisionTreeClassifier): Model for test set evaluation (initialized as sentinel).
        acceptable_gap (float): Minimum gap for acceptable generalization (default: DEFAULT_ACCEPTABLE_GAP).
        large_gap (float): Minimum gap indicating overfitting (default: DEFAULT_LARGE_GAP).        

    Example:
        >>> dtree = DecisionTree(
        ...     data_splits=splits,
        ...     cv=5,
        ...     param_grid={'max_depth': [None, 10, 20],
        ...                 'min_samples_split': [2, 5, 10]},
        ...     class_weight='balanced',
        ...     n_iter=10
        ... )
        >>> dtree.tune_hyperparameters()  # Workflow methods update attributes
        >>> dtree.validate_model(ModelBalancing.Auto_Balanced)

    Note:
        Attributes are initialized with sentinel values and are updated by calling
        workflow methods in sequence: tune_hyperparameters() → validate_model() → etc.
        The tune_hyperparameters() method automatically chooses RandomizedSearchCV for
        scipy distributions or GridSearchCV for discrete lists.
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
            n_jobs: int = -1,
            n_iter: int = 10,
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
        self.hyperparameters = DecisionTreeHyperParameters.empty()
        self.validation_results = ValidationResults.empty()
        self.validation_model_feature_importances = ModelFeatureImportance.empty(
            ModelType.Decision_Tree)
        self.test_model = DecisionTreeClassifier()

    @classmethod
    def model_from_hyperparameters(
        cls,
        params: dict,
        random_state: int,
        class_weight: Optional[Union[
            Mapping[str, float],
            str
        ]],
    ) -> DecisionTreeClassifier:
        """Create a DecisionTreeClassifier from hyperparameters dictionary.

        Factory method that constructs a configured DecisionTreeClassifier instance
        using hyperparameters from a dictionary. The returned classifier is configured
        but not yet trained.

        Args:
            params (dict): Hyperparameters with keys 'max_depth', 'min_samples_leaf', 'min_samples_split'.
            random_state (int): Random seed for reproducibility.
            class_weight (Optional[Union[Mapping[str, float], str]]): Class balancing strategy.
                Can be 'balanced', a dict mapping class labels to weights, or None.

        Returns:
            DecisionTreeClassifier: Configured but untrained classifier.

        Example:
            >>> params = {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5}
            >>> model = DecisionTree.model_from_hyperparameters(params, random_state=42, class_weight='balanced')
            >>> model.fit(X_train, y_train)
        """
        return DecisionTreeClassifier(
            random_state=random_state,
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            min_samples_split=params['min_samples_split'],
            class_weight=class_weight
        )

    @classmethod
    def optimal_threshold(
        cls,
        data_splits: DataSplits,
        model: DecisionTreeClassifier,
    ) -> tuple[float, float, np.ndarray]:
        """Find optimal classification threshold by maximizing F1 score on validation set.

        Trains the model, generates probability predictions on the validation set,
        and finds the threshold that maximizes F1 score using precision-recall curve.
        Then evaluates the model on the test set using this optimal threshold.

        Args:
            data_splits (DataSplits): Training, validation, and test data splits.
            model (DecisionTreeClassifier): Untrained decision tree classifier to optimize.

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
        valid_proba_array = model.predict_proba(data_splits.features_valid)
        valid_proba = valid_proba_array[:, 1]  # type: ignore[misc]

        precisions, recalls, thresholds = precision_recall_curve(
            data_splits.target_valid,
            valid_proba
        )

        f1_scores = np.nan_to_num(
            2 * (precisions * recalls) / (precisions + recalls))

        optimal_threshold_index = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_threshold_index]

        test_proba_array = model.predict_proba(data_splits.features_test)
        test_proba = test_proba_array[:, 1]  # type: ignore[misc]
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

        Trains multiple decision tree models with different hyperparameter combinations
        using cross-validation on the training set.

        Note:
            Updates hyperparameters attribute.
            Uses RandomizedSearchCV if param_grid contains scipy distributions, otherwise
            uses GridSearchCV for discrete parameter lists.
        """
        search_cv_model = DecisionTreeClassifier(
            random_state=self.random_state,
            class_weight=self.class_weight
        )

        # Automatically detect whether to use RandomizedSearchCV or GridSearchCV
        # by checking if param_grid contains scipy distribution objects
        use_randomized = any(
            hasattr(value, 'rvs')  # scipy distributions have .rvs() method
            for value in self.param_grid.values()
        )

        if use_randomized:
            # Use RandomizedSearchCV for distributions (e.g., sp_randint)
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
        self.hyperparameters = DecisionTreeHyperParameters.from_search_cv(
            search_cv)

    def validate_model(
            self,
            model_balancing: ModelBalancing,
            header: str,
            include_in_report: bool = True,
    ) -> tuple[ModelConfiguration, str]:
        """Train model with best hyperparameters and evaluate on validation set.

        Uses the optimal hyperparameters from GridSearchCV to train a new model
        and evaluates its performance on the training and validation sets.

        Args:
            model_balancing (ModelBalancing): The balancing strategy applied to data.
            include_in_report (bool): Whether to print validation results and feature 
                importances.
            header (str): Display header for report.

        Returns:
            ModelConfiguration: Configuration object with validation metrics.

        Note:
            Updates validation_results, hyperparameters (with train/valid F1 scores).
        """
        report_text = ''

        validation_model = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=self.hyperparameters.max_depth,
            min_samples_leaf=self.hyperparameters.min_samples_leaf,
            min_samples_split=self.hyperparameters.min_samples_split,
            class_weight=self.class_weight
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

        self.hyperparameters = DecisionTreeHyperParameters(
            max_depth=self.hyperparameters.max_depth,
            min_samples_leaf=self.hyperparameters.min_samples_leaf,
            min_samples_split=self.hyperparameters.min_samples_split,
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

        self.validation_model_feature_importances = ModelFeatureImportance(
            features=self.data_splits.features_train.columns.tolist(),
            model=validation_model
        )

        if include_in_report:
            report_text += self.hyperparameters.info(header=header)

        optimized_params = {
            'max_depth': self.hyperparameters.max_depth,
            'min_samples_leaf': self.hyperparameters.min_samples_leaf,
            'min_samples_split': self.hyperparameters.min_samples_split
        }

        return (ModelConfiguration(
            model_type=ModelType.Decision_Tree,
            model_balancing=model_balancing,
            evaluation_method=ModelEvaluationMethod.Grid_Search,
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

    def calc_best_top_n(
            self,
            feature_importance: ModelFeatureImportance,
            model_balancing: ModelBalancing,
    ) -> tuple[list[ModelConfiguration], ModelConfiguration, str]:
        """Find optimal number of top features for best model performance.

        Tests models with varying numbers of most important features to identify
        the feature subset that maximizes validation F1 score.

        Args:
            feature_importance (ModelFeatureImportance): Feature importance rankings to use.
            model_balancing (ModelBalancing): The balancing strategy for data.

        Returns:
            tuple: (list of all model configurations, best configuration)
        """
        report_text = ''

        column_names = [
            'Feature Set',
            'CV F1 Score',
            'Training F1 Score',
            'Validation F1 Score',
            'Model Generalization',
            'max_depth',
            'min_samples_leaf',
            'min_samples_split'
        ]

        top_n_df = pd.DataFrame(columns=column_names)
        feature_importance.calc_threshold_indices()
        best_f1_score_valid = 0.0
        best_n = 0
        best_mc = ModelConfiguration.empty()
        mc_list: list[ModelConfiguration] = []

        for n in range(feature_importance.threshold_80_idx, feature_importance.threshold_95_idx + 1):
            top_n_features = feature_importance.features[0:n]

            # Use memory-efficient feature subset method instead of creating new DataSplits
            data_splits = self.data_splits.with_feature_subset(top_n_features)

            top_n_dtree = DecisionTree(
                data_splits=data_splits,
                cv=self.cv,
                param_grid=self.param_grid,
                class_weight=self.class_weight,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                n_iter=self.n_iter
            )

            top_n_dtree.tune_hyperparameters()

            mc, t = top_n_dtree.validate_model(
                model_balancing=model_balancing,
                include_in_report=False,
                header=f'Decision Tree - Top {n} Features'
            )

            report_text += t
            mc_list.append(mc)
            f1_score_train = 0.0
            f1_score_valid = 0.0

            if isinstance(top_n_dtree.hyperparameters.f1_score_train, float):
                f1_score_train = top_n_dtree.hyperparameters.f1_score_train

            if isinstance(top_n_dtree.hyperparameters.f1_score_valid, float):
                f1_score_valid = top_n_dtree.hyperparameters.f1_score_valid

            if f1_score_valid > best_f1_score_valid:
                best_f1_score_valid = f1_score_valid
                best_n = n
                best_mc = mc

            top_n_result = [
                f'Top {n}',
                top_n_dtree.hyperparameters.f1_score_cv,
                f1_score_train,
                f1_score_valid,
                top_n_dtree.hyperparameters.model_generalization.name,
                top_n_dtree.hyperparameters.max_depth,
                top_n_dtree.hyperparameters.min_samples_leaf,
                top_n_dtree.hyperparameters.min_samples_split
            ]

            top_n_df.loc[n] = top_n_result

        f1_formatters: dict[str | int, Any] = {
            'CV F1 Score': lambda x: f'{x:{F1_FORMAT}}',
            'Training F1 Score': lambda x: f'{x:{F1_FORMAT}}',
            'Validation F1 Score': lambda x: f'{x:{F1_FORMAT}}'
        }

        best_row = top_n_df.loc[best_n]

        report_text += f'''

{'Feature Importance'.center(REPORT_WIDTH, '-')}
{top_n_df.to_string(formatters=f1_formatters)}

Best N: {best_n}
Best Validation F1 Score:
Feature Set            {best_row['Feature Set']}
CV F1 Score            {best_row['CV F1 Score']:{F1_FORMAT}}
Training F1 Score      {best_row['Training F1 Score']:{F1_FORMAT}}
Validation F1 Score    {best_row['Validation F1 Score']:{F1_FORMAT}}
Model Generalization   {best_row['Model Generalization']}
max_depth              {best_row['max_depth']}
min_samples_leaf       {best_row['min_samples_leaf']}
min_samples_split      {best_row['min_samples_split']}
'''

        return (mc_list, best_mc, report_text)

    def evaluate_model(
            self,
            model_balancing: ModelBalancing,
            perform_feature_selection: bool,
            is_first_model: bool = False,
            include_in_report: bool = True,
    ) -> tuple[list[ModelConfiguration], str]:
        """Complete model evaluation pipeline: search, validate, and optimize features.

        Executes the full workflow: GridSearchCV for hyperparameters, validation
        with best params, and feature selection to find optimal feature subset.

        Args:
            model_balancing (ModelBalancing): The balancing strategy for data.
            perform_feature_selection (bool): Whether calc_best_top_n should be called. The
                feature importance data is created in the call to validate_model(). This
                parameter determines whether each of the best N values is evaluated
                individually, which involves a call to a CV function.
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
            header=f'Decision Tree ({model_balancing.name} - All Features)'
        )

        mc_list.append(mc)
        report_text += t

        if perform_feature_selection:
            top_n_mc_list, _, t = self.calc_best_top_n(
                feature_importance=self.validation_model_feature_importances,
                model_balancing=model_balancing
            )

            mc_list.extend(top_n_mc_list)
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
            model_type=ModelType.Decision_Tree,
            model_balancing=model_balancing,
            params=self.param_grid,
            df=self.data_splits.target_valid,
            include_in_report=include_in_report
        )

        mc_list.append(mc)
        report_text += t

        return mc_list, report_text

    def prepare_test_model(
            self,
            model_configuration: ModelConfiguration,
            include_in_report: bool = True,
            header: str = 'Decision Tree (Validation Set)'
    ) -> str:
        """Train final model on combined train+validation data for test evaluation.

        Args:
            model_configuration (ModelConfiguration): The corresponding model configuration.
            include_in_report (bool): Whether to print test set hyperparameters.
            header (str): Display header for report.

        Note:
            Trains on combined train+valid sets to maximize training data before test.
        """
        report_text = ''

        self.test_model = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=self.param_grid['max_depth'],
            min_samples_leaf=self.param_grid['min_samples_leaf'],
            min_samples_split=self.param_grid['min_samples_split'],
            class_weight=model_configuration.class_weight
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

        dtree = DecisionTree(
            data_splits=model_configuration.data_splits,
            cv=model_configuration.cv,
            param_grid=model_configuration.params,
            class_weight=model_configuration.class_weight,
            scoring=model_configuration.scoring,
            n_jobs=model_configuration.n_jobs,
            n_iter=model_configuration.n_iter
        )

        text_result += dtree.prepare_test_model(
            model_configuration=model_configuration,
            include_in_report=True,
            header='Decision Tree - Best Model (Validation Set)'
        )

        predicted_test = dtree.test_model.predict(
            model_configuration.data_splits.features_test)
        f1_result = cast(float, f1_score(
            model_configuration.data_splits.target_test, predicted_test))
        text_result += f'Test set F1 score: {f1_result:{F1_FORMAT}}'

        return f1_result, text_result
