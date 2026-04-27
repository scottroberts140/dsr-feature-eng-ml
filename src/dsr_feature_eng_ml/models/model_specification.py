"""Base model specification and shared training utilities."""

from __future__ import annotations

import dataclasses
import enum
import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Mapping,
    Optional,
    Protocol,
    Self,
    Tuple,
    Type,
    TypeGuard,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import psutil
from dsr_utils.formatting import DataScale, EnumFormat
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)

if TYPE_CHECKING:
    from dsr_feature_eng_ml.evaluation.schema import (
        DataSplits,
        FeatureMetadata,
        ModelConfiguration,
        ModelFeatureImportance,
    )

from dsr_feature_eng_ml.prefs_instance import prefs


@runtime_checkable
class ScikitModel(Protocol):
    """
    Protocol defining the minimal interface for scikit-learn compatible estimators.

    This protocol ensures that any object passed to the Auditor or ModelSpecification
    supports the fundamental fit/predict/params API required for machine learning
    workflows.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Self:
        """
        Train the model on the provided data.

        Parameters
        ----------
        X : Any
            The input features (array-like, DataFrame, or sparse matrix).
        y : Any
            The target labels or values.
        sample_weight : Any, optional
            Individual weights for each training sample.

        Returns
        -------
        Self
            The fitted estimator instance.
        """
        ...

    def predict(self, X: Any) -> Any:
        """
        Generate predictions for the input samples.

        Parameters
        ----------
        X : Any
            The input features to predict.

        Returns
        -------
        Any
            Predicted values or class labels.
        """
        ...

    def get_params(self, deep: bool = True) -> Mapping[str, Any]:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        Mapping[str, Any]
            Parameter names mapped to their values.
        """
        ...


@runtime_checkable
class ProbabilisticClassifier(ScikitModel, Protocol):
    """
    Protocol for classifiers that support probability estimation.

    Extends ScikitModel to include the predict_proba method, required for
    metrics like ROC-AUC and Log Loss.
    """

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : Any
            The input features.

        Returns
        -------
        np.ndarray
            The class probabilities of the input samples.
        """
        ...


# T_Params ensures that hyperparameter containers are subclasses of ModelParams
T_Params = TypeVar("T_Params", bound="ModelParams")

# T_Estimator ensures that the internal model instance satisfies the ScikitModel Protocol
T_Estimator = TypeVar("T_Estimator", bound="ScikitModel")


def calculate_search_iterations(
    grid: dict[str, Any],
    min_iter: int = -1,
    max_iter: int = 100,
    coverage: float = 0.10,
) -> int:
    """
    Calculate n_iter based on a percentage of the total search space.

    Parameters
    ----------
    grid : dict
        The hyperparameter grid or distribution dictionary.
    min_iter : int, default -1
        The absolute minimum iterations allowed.
    max_iter : int, default 100
        The absolute maximum iterations allowed.
    coverage : float, default 0.10
        The target percentage of the search space to cover.

    Returns
    -------
    int
        The suggested number of iterations for a RandomSearch.
    """
    import math

    # 1. Calculate the total size of the grid
    total_combinations = 1
    for values in grid.values():
        # If it's a distribution (SciPy rvs), it has infinite size.
        if hasattr(values, "rvs"):
            total_combinations = int(max_iter / coverage)
            break

        if isinstance(values, (list, tuple)):
            total_combinations *= len(values)

    # 2. Apply coverage
    calculated_iter = int(math.ceil(total_combinations * coverage))

    # 3. Ensure a floor for small grids and cap at max_iter
    actual_iter = max(calculated_iter, min(total_combinations, 5))
    return max(min_iter, min(actual_iter, max_iter))


@dataclass(frozen=True)
class ModelParams(ABC):
    """
    Abstract base dataclass for model-specific hyperparameters.

    Acts as an immutable container for model settings. Subclasses define
    specific parameters (e.g., n_estimators, max_depth) while inheriting
    standardized serialization and search space calculations.

    Attributes
    ----------
    random_state : int, optional
        Seed used for reproducibility. Defaults to None.
    optimization_strategy : OptimizationStrategy
        The method used to select these parameters (Manual, Grid, or Random).
    """

    random_state: Optional[int] = None
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL

    @abstractmethod
    def info(self) -> str:
        """Return a human-readable summary of the parameters."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Convert parameters to a dictionary suitable for scikit-learn.

        Automatically filters out None values and converts Enum members
        to their name strings for compatibility.

        Returns
        -------
        dict[str, Any]
            A mapping of parameter names to values.
        """
        data = {}
        # dataclasses.asdict provides a deep copy of the fields
        for k, v in dataclasses.asdict(self).items():
            if v is None:
                continue

            # Standardize Enum serialization (e.g., OptimizationStrategy.MANUAL -> "MANUAL")
            if isinstance(v, enum.Enum):
                data[k] = v.name
            else:
                data[k] = v

        return data

    @property
    def num_candidates(self) -> int:
        """The total number of parameter combinations to be evaluated."""
        if self.optimization_strategy == OptimizationStrategy.MANUAL:
            return 1

        params_dict = self.to_dict()

        if self.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            return calculate_search_iterations(params_dict, min_iter=-1)

        # GRID_SEARCH Logic...
        import math

        nc = math.prod(
            [
                len(v) if isinstance(v, (list, tuple)) else 1
                for v in params_dict.values()
            ]
        )
        return max(1, nc)

    @staticmethod
    @abstractmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Define the default hyperparameter search space for this model type.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns a smaller, more efficient search grid.
        """
        pass


@dataclass(frozen=True)
class ClassificationModelParams(ModelParams, ABC):
    """Base hyperparameter class for classification-only models."""

    task_type: TaskType = TaskType.CLASSIFICATION
    scoring: ScoringMetric = ScoringMetric.F1


@dataclass(frozen=True)
class RegressionModelParams(ModelParams, ABC):
    """Base hyperparameter class for regression-only models."""

    task_type: TaskType = TaskType.REGRESSION
    scoring: ScoringMetric = ScoringMetric.R2


class ModelSpecification(ABC, Generic[T_Params, T_Estimator]):
    """
    Abstract base class defining the lifecycle and configuration of an ML model.

    This class provides the infrastructure for hyperparameter tuning, cross-validation,
    and performance auditing. It cannot be instantiated directly; model-specific
    subclasses (e.g., RandomForestRegressor) must implement the abstract methods.

    Attributes
    ----------
    cv : int
        Number of cross-validation folds.
    balancing_strategy : BalancingStrategy
        Strategy for handling class imbalance (Classification only).
    n_jobs : int
        Number of parallel processes to use during training.
    n_iter : int
        Number of iterations for randomized search. If -1, calculated automatically.
    max_iter : int
        Maximum iterations for iterative estimators (e.g., Logistic Regression).
    acceptable_gap : float
        The maximum training-to-validation gap for a 'Well-Fit' status.
    large_gap : float
        The threshold above which a model is considered 'Overfit'.
    predicted_train : pd.Series
        Cached predictions on the training set after evaluation.
    predicted_val : pd.Series
        Cached predictions on the validation set after evaluation.
    optimization_strategy : OptimizationStrategy
        The method used for parameter tuning (MANUAL, GRID, or RANDOM).
    """

    params_class: Type[T_Params]

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[T_Params] = None,
        n_jobs: int = 3,
        n_iter: int = -1,
        max_iter: int = 1000,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the shared model configuration."""
        self.cv = cv
        self.balancing_strategy = balancing_strategy

        # Validation: Ensure the scoring metric is compatible with the task type
        valid_metrics = self.get_valid_scoring_metrics()
        if self.scoring not in valid_metrics:
            enum_format = EnumFormat(use_value=False)
            raise ValueError(
                f"Invalid metric '{enum_format.format_value(self.scoring)}' "
                f"for {self.task_type.value}. Valid: {[m.name for m in valid_metrics]}"
            )

        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.acceptable_gap = acceptable_gap
        self.large_gap = large_gap
        self.optimization_strategy = optimization_strategy

        # Managed Mutability placeholders
        self.predicted_train = pd.Series(dtype=float)
        self.predicted_val = pd.Series(dtype=float)
        self.estimator: Optional[T_Estimator] = None

    @abstractmethod
    def get_estimator_class(self) -> Type[T_Estimator]:
        """Return the scikit-learn estimator class reference."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """The high-level ML task (Classification or Regression)."""
        pass

    @property
    @abstractmethod
    def scoring(self) -> ScoringMetric:
        """The metric used to optimize and evaluate the model."""
        pass

    @scoring.setter
    @abstractmethod
    def scoring(self, value: ScoringMetric) -> None:
        """Set the optimization metric."""
        pass

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """The specific enum identifier for this model (e.g., RIDGE)."""
        pass

    @property
    @abstractmethod
    def model_dials(self) -> T_Params:
        """The hyperparameter container for this specific model."""
        pass

    @model_dials.setter
    @abstractmethod
    def model_dials(self, value: T_Params) -> None:
        """Update the hyperparameter configuration."""
        pass

    @property
    def num_candidates(self) -> int:
        """Total hyperparameter combinations to be evaluated."""
        return self.model_dials.num_candidates

    @property
    def total_fits(self) -> int:
        """
        The total number of estimator training cycles (candidates * CV folds).
        """
        cv_count = self.cv if self.cv is not None else 5

        if self.optimization_strategy == OptimizationStrategy.MANUAL:
            return cv_count

        nc = self.num_candidates
        # Fallback to n_iter if num_candidates logic hasn't resolved yet
        if nc is None or nc <= 0:
            nc = self.n_iter if self.n_iter != -1 else 1

        return cv_count * nc

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """
        Extract importance/coefficients from the fitted estimator.

        Returns
        -------
        Optional[np.ndarray]
            An array of feature weights, normalized as absolute values
            for linear models. Returns None if model is not fitted.
        """
        if self.estimator is None:
            return None

        # 1. Check for standard tree-based importance
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is not None:
            return cast(np.ndarray, importances)

        # 2. Check for linear coefficients
        coef = getattr(self.estimator, "coef_", None)
        if coef is not None:
            # Flatten in case of multi-class coefficients
            return np.abs(cast(np.ndarray, coef)).flatten()

        return None

    @abstractmethod
    def create_estimator(self, parameters: Optional[T_Params] = None) -> T_Estimator:
        """Instantiate a raw scikit-learn estimator with current settings."""
        pass

    def get_valid_scoring_metrics(self) -> list[ScoringMetric]:
        """Return valid scoring metrics for this model specification."""
        return ScoringMetric.get_valid_metrics(self.task_type)

    def resolve_search_scoring_key(self) -> str:
        """Return the scoring key used for CV search."""
        return self.scoring.value

    def is_probabilistic(self, estimator: Any) -> TypeGuard[ProbabilisticClassifier]:
        """
        Verify if the estimator supports class probability prediction.
        """
        return hasattr(estimator, "predict_proba")

    def _get_fitted_estimator(self) -> T_Estimator:
        """
        Internal helper to ensure the estimator exists before use.

        Raises
        ------
        RuntimeError
            If called before the model has been fitted.
        """
        if self.estimator is None:
            raise RuntimeError(
                f"The {self.model_type.value} estimator has not been fitted yet. "
                "Call 'fit()' or 'tune()' before attempting to generate predictions."
            )
        return self.estimator

    def tune_model(
        self,
        data_splits: DataSplits,
        method: OptimizationStrategy,
        features_to_fit_set: set[FeatureMetadata],
        custom_grid: Optional[dict[str, Any]] = None,
        use_combined_data: bool = False,
        max_sample_size: Optional[int] = None,
        perform_memory_check: bool = True,
    ) -> Tuple[T_Params, float, bool, float, float, float, float]:
        """
        Execute hyperparameter optimization and update model parameters.

        This method performs a GridSearchCV or RandomizedSearchCV on a subset of the
        data to find optimal hyperparameters while preventing system OOM (Out of Memory)
        errors through predictive memory auditing.

        Parameters
        ----------
        data_splits : DataSplits
            Container for training and validation datasets.
        method : OptimizationStrategy
            The base search strategy (GRID_SEARCH or RANDOM_SEARCH).
        features_to_fit_set : set[FeatureMetadata]
            The set of features to include in the training process.
        custom_grid : dict[str, Any], optional
            A user-provided search space. If None, the standard library grid is used.
        use_combined_data : bool, default False
            If True, combines training and validation sets for the tuning phase.
        max_sample_size : int, optional
            The maximum number of rows to use for tuning. If None, all available
            rows are used.
        perform_memory_check : bool, default True
            If True, validates memory headroom before starting the search.

        Returns
        -------
        best_params : T_Params
            The updated hyperparameter container with the best found values.
        best_score : float
            The mean cross-validated score of the best_estimator.
        risk_triggered : bool
            True if the memory audit detected a high risk of OOM.
        available_gb : float
            System memory available at the start of tuning (in GB).
        estimated_peak_gb : float
            Predicted maximum memory usage for the tuning process (in GB).
        model_multiplier : float
            The complexity multiplier used for the memory estimation.
        sampling_factor : float
            The percentage of the dataset actually used (e.g., 0.1 for 10%).
        """
        # 1. Prepare Data
        if use_combined_data:
            features = pd.concat([data_splits.train_features, data_splits.val_features])
            target = pd.concat([data_splits.train_target, data_splits.val_target])
        else:
            features = data_splits.train_features
            target = data_splits.train_target

        feature_list = [f.name for f in features_to_fit_set]
        features = features[feature_list]
        total_rows = len(features)

        # Initialize monitoring variables
        memory_risk_triggered = False
        estimated_peak_gb, available_gb = 0.0, 0.0
        model_multiplier, sampling_factor = 1.0, 1.0

        # Internal Helper: Sampling
        def _apply_sampling(size: int) -> Tuple[pd.DataFrame, pd.Series[Any], float]:
            if size >= total_rows:
                return features, target, 1.0

            s_feat = features.sample(n=size, random_state=data_splits.random_state)
            s_targ = target.loc[s_feat.index]
            factor = size / total_rows
            return s_feat, s_targ, factor

        # 2. Optional Sampling (Speed)
        # Respect explicit caller cap. If no cap is provided, tune on all rows.
        if max_sample_size is None:
            max_sample_size = total_rows

        if total_rows > max_sample_size:
            print(f"⚠️ Dataset ({total_rows:,} rows) exceeds tuning safety limit.")
            tuning_features, tuning_target, sampling_factor = _apply_sampling(
                max_sample_size
            )
            print(
                f"📉 Sampling {len(tuning_features):,} rows ({sampling_factor:.1%}) for optimization..."
            )
        else:
            tuning_features, tuning_target = features, target

        # 3. Memory Safety Check
        if perform_memory_check:
            from dsr_feature_eng_ml.utils.memory import check_memory_risk

            memory_risk_triggered, estimated_peak_gb, available_gb, model_multiplier = (
                check_memory_risk(tuning_features, self, self.n_jobs)
            )

        # 4. Emergency Downsampling (OOM Prevention)
        if memory_risk_triggered:
            print(
                f"🚨 DANGER: Predicted peak {prefs.gb_format.format_value(estimated_peak_gb)} "
                f"exceeds safety buffer of available {prefs.gb_format.format_value(available_gb)}."
            )

            # Reduce sample size based on the ratio of available vs required memory
            safety_limit = int(
                len(tuning_features) * (available_gb / estimated_peak_gb) * 0.7
            )
            tuning_features, tuning_target, sampling_factor = _apply_sampling(
                safety_limit
            )
            print(
                f"📉 Emergency downsampling to {len(tuning_features):,} rows ({sampling_factor:.1%})..."
            )

        # 5. Build Search Space
        if custom_grid:
            grid = custom_grid
        else:
            # Detect if current dials already contain a search space (lists/tuples)
            current_dials_dict = dataclasses.asdict(self.model_dials)
            search_space = {
                k: v
                for k, v in current_dials_dict.items()
                if isinstance(v, (list, tuple))
            }
            grid = (
                search_space
                if search_space
                else self.params_class.get_standard_search_grid(narrow=True)
            )

        # Determine Strategy
        is_dist_search = any(hasattr(v, "rvs") for v in grid.values())
        if is_dist_search or method == OptimizationStrategy.RANDOM_SEARCH:
            self.optimization_strategy = OptimizationStrategy.RANDOM_SEARCH
        else:
            self.optimization_strategy = OptimizationStrategy.GRID_SEARCH

        # 6. Execute Search
        base_estimator = self.create_estimator()

        def _prepare_search_grid(est: Any, raw_grid: dict) -> dict:
            valid_params = est.get_params().keys()
            clean_grid = {}
            for k, v in raw_grid.items():
                # Skip unset knobs; passing None through CV search can override
                # estimator defaults and trigger deprecation/inconsistency warnings.
                if v is None:
                    continue

                # sklearn LogisticRegression deprecates explicit 'penalty'.
                # We configure it through l1_ratio/C in the estimator factory.
                if est.__class__.__name__ == "LogisticRegression" and k == "penalty":
                    continue

                if k in valid_params:
                    # CV Searchers require iterables or distributions
                    clean_grid[k] = (
                        v if isinstance(v, (list, tuple)) or hasattr(v, "rvs") else [v]
                    )
            return clean_grid

        if self.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            refined_grid = _prepare_search_grid(base_estimator, grid)
            if self.n_iter == -1:
                self.n_iter = calculate_search_iterations(refined_grid)

            scoring_key = self.resolve_search_scoring_key()

            search_cv = RandomizedSearchCV(
                estimator=cast(BaseEstimator, base_estimator),
                param_distributions=refined_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=scoring_key,
                n_jobs=self.n_jobs,
                verbose=prefs.cv_verbose,
                random_state=data_splits.random_state,
            )
        else:
            scoring_key = self.resolve_search_scoring_key()

            search_cv = GridSearchCV(
                estimator=cast(BaseEstimator, base_estimator),
                param_grid=_prepare_search_grid(base_estimator, grid),
                cv=self.cv,
                scoring=scoring_key,
                n_jobs=self.n_jobs,
                verbose=prefs.cv_verbose,
            )

        search_cv.fit(tuning_features, tuning_target)

        # 7. Update State and Return
        self.model_dials = dataclasses.replace(
            self.model_dials, **search_cv.best_params_
        )

        return (
            self.model_dials,
            search_cv.best_score_,
            memory_risk_triggered,
            available_gb,
            estimated_peak_gb,
            model_multiplier,
            sampling_factor,
        )

    def fit(
        self,
        data_splits: DataSplits,
        features_to_fit_set: set[FeatureMetadata],
        use_combined_data: bool = False,
    ) -> Tuple[float, float]:
        """
        Train the estimator and capture memory performance metrics.

        This method handles data balancing (resampling or weighting), instantiates
        the raw estimator, and measures the RSS (Resident Set Size) memory delta
        specifically during the fitting process.

        Parameters
        ----------
        data_splits : DataSplits
            The container for training and validation datasets.
        features_to_fit_set : set[FeatureMetadata]
            Metadata defining which columns to include in the training set.
        use_combined_data : bool, default False
            If True, training and validation sets are merged before fitting.

        Returns
        -------
        mem_used : float
            The difference in memory usage (GB) between start and end of fit.
        peak_rss : float
            The absolute peak Resident Set Size (GB) measured after fit.
        """
        # 1. Prepare Data based on Balancing Strategy
        X, y = data_splits.get_balanced_train_data(
            strategy=self.balancing_strategy,
            feature_set=features_to_fit_set,
            use_combined_data=use_combined_data,
        )

        # 2. Extract weights for WEIGHTED strategy
        weights = data_splits.get_train_weights(
            self.balancing_strategy,
            is_regression=(self.task_type == TaskType.REGRESSION),
        )

        # 3. Instantiate and Fit
        self.estimator = self.create_estimator()

        # Memory baseline
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        # Execute fit. Not all sklearn estimators support sample_weight
        # (e.g., KNeighborsClassifier), so pass it only when accepted.
        fit_sig = inspect.signature(self.estimator.fit)
        accepts_sample_weight = "sample_weight" in fit_sig.parameters
        if accepts_sample_weight and weights is not None:
            self.estimator.fit(X, y, sample_weight=weights)
        else:
            self.estimator.fit(X, y)

        # Memory telemetry
        mem_after = process.memory_info().rss
        mem_used = mem_after - mem_before

        return DataScale.GB.get_scaled_value(
            float(mem_used)
        ), DataScale.GB.get_scaled_value(float(mem_after))

    def fit_and_evaluate_val(
        self,
        data_splits: DataSplits,
        id: str,
        features_to_fit_set: set[FeatureMetadata],
        score_cv: Optional[float] = None,
        use_combined_data: bool = False,
        filter_outliers: bool = False,
        outlier_count: int = prefs.default_worst_errors_n,
    ) -> ModelConfiguration:
        """
        Execute a full training cycle and generate a validation-scored configuration.

        This method encapsulates the "Fit" and "Analyze" workflow: it trains the
        underlying estimator, captures telemetry, and then triggers the evaluation
        logic to compute generalization metrics and store predictions.

        Parameters
        ----------
        data_splits : DataSplits
            The dataset container providing training and validation splits.
        id : str
            A unique identifier for this specific model configuration.
        features_to_fit_set : set[FeatureMetadata]
            The specific set of features used for training.
        score_cv : float, optional
            The cross-validation score obtained during tuning, if applicable.
        use_combined_data : bool, default False
            Whether the model was trained on the combined train/validation sets.
        filter_outliers : bool, default False
            If True, calculates a "cleaned" set of metrics by removing the
            observations with the largest errors.
        outlier_count : int, default prefs.default_worst_errors_n
            The number of high-error observations to exclude if filter_outliers is True.

        Returns
        -------
        ModelConfiguration
            A populated container holding the fitted estimator parameters,
            performance metrics, and system telemetry (memory/time).
        """
        # 1. Action: Fit the model and capture memory telemetry
        mem_used, mem_peak = self.fit(
            data_splits=data_splits,
            features_to_fit_set=features_to_fit_set,
            use_combined_data=use_combined_data,
        )

        # 2. Analysis: Generate metrics and compile the ModelConfiguration
        # Note: self.evaluate_val_performance internally calls calc_predictions()
        return self.evaluate_val_performance(
            data_splits=data_splits,
            id=id,
            features_to_fit_set=features_to_fit_set,
            mem_used=mem_used,
            mem_peak=mem_peak,
            score_cv=score_cv,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
            use_combined_data=use_combined_data,
        )

    def _score_classification(
        self,
        features: pd.DataFrame,
        targets: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> Tuple[float, Optional[float], pd.Series, pd.DataFrame]:
        """
        Compute weighted F1 scores and extract class probabilities.

        This internal method handles the logic for calculating baseline performance
        and the "cleaned" performance metric, which excludes the most confident
        incorrect predictions (outliers).

        Parameters
        ----------
        features : pd.DataFrame
            The feature matrix used for generating predictions.
        targets : pd.Series
            The ground-truth class labels.
        filter_outliers : bool
            If True, calculates an additional 'cleaned' F1 score.
        outlier_count : int
            The number of high-confidence errors to exclude from the cleaned score.

        Returns
        -------
        f1 : float
            The standard weighted F1 score.
        f1_cleaned : float, optional
            The weighted F1 score after removing outliers. None if filter_outliers is False.
        preds : pd.Series
            Discrete class predictions indexed to the input features.
        probs : pd.DataFrame
            Class probabilities indexed to the input features.
        """
        # 1. Safety Guard: Ensure estimator is fitted and supports probabilities
        # This solves the Pylance "estimator is None" and "predict_proba missing" errors
        estimator = self._get_fitted_estimator()

        # We cast to ProbabilisticClassifier to ensure Pylance recognizes predict_proba
        prob_estimator = cast(ProbabilisticClassifier, estimator)

        # 2. Generate raw outputs
        raw_preds = prob_estimator.predict(features)
        raw_probs = prob_estimator.predict_proba(features)

        # 3. Align with original indices
        preds = pd.Series(raw_preds, index=targets.index, name="predictions")
        probs = pd.DataFrame(raw_probs, index=targets.index)

        # 4. Standard Metric Calculation
        from sklearn.metrics import f1_score

        f1 = float(f1_score(targets, preds, average="weighted"))

        # 5. Outlier Filtering (Confident Mistakes)
        f1_cleaned: Optional[float] = None

        if filter_outliers:
            # Mask where prediction is wrong
            incorrect_mask = targets.to_numpy() != raw_preds

            # Confidence is the max probability assigned to any single class
            confidences = np.max(raw_probs, axis=1)

            # Assign high scores only to incorrect predictions
            # Correct predictions are ignored (set to -1)
            error_scores = np.where(incorrect_mask, confidences, -1.0)

            # Partition to find indices of the worst errors
            n_to_drop = min(
                outlier_count, int(len(error_scores) * 0.5)
            )  # Safety cap at 50%
            n_to_keep = len(error_scores) - n_to_drop

            # Identify indices of the observations to retain
            keep_indices = np.argpartition(error_scores, n_to_keep)[:n_to_keep]

            f1_cleaned = float(
                f1_score(
                    targets.iloc[keep_indices],
                    raw_preds[keep_indices],
                    average="weighted",
                )
            )

        return f1, f1_cleaned, preds, probs

    def _score_regression(
        self,
        features: pd.DataFrame,
        targets: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> tuple[float, float, float, Optional[float], pd.Series]:
        """
        Compute standard regression metrics and residual-based 'cleaned' R2.

        Parameters
        ----------
        features : pd.DataFrame
            The feature matrix for generating predictions.
        targets : pd.Series
            The ground-truth numerical targets.
        filter_outliers : bool
            If True, calculates an additional 'cleaned' R2 score.
        outlier_count : int
            The number of high-absolute-error observations to exclude.

        Returns
        -------
        mae : float
            Mean Absolute Error.
        mse : float
            Mean Squared Error.
        r2 : float
            Coefficient of Determination (R-squared).
        r2_cleaned : float, optional
            R-squared calculated after removing outlier residuals.
        preds : pd.Series
            Predictions indexed to the input features.
        """
        # 1. Safety Guard: Ensure estimator exists and is fitted
        estimator = self._get_fitted_estimator()

        # 2. Generate predictions and align indices
        raw_preds = estimator.predict(features)
        preds = pd.Series(raw_preds, index=targets.index, name="predictions")

        # 3. Standard Metric Calculation
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = float(mean_absolute_error(targets, preds))
        mse = float(mean_squared_error(targets, preds))
        r2 = float(r2_score(targets, preds))

        # 4. Outlier Filtering (Largest Residuals)
        r2_cleaned: Optional[float] = None

        if filter_outliers:
            # Calculate absolute residuals
            abs_errors = np.abs((targets.to_numpy() - raw_preds).flatten())

            # Efficiency: O(n) partition to find the top N errors
            n_to_drop = min(outlier_count, int(len(abs_errors) * 0.5))
            n_to_keep = len(abs_errors) - n_to_drop

            # Retrieve indices of the samples with the lowest errors
            keep_indices = np.argpartition(abs_errors, n_to_keep)[:n_to_keep]

            # Calculate the "Cleaned" version of the score
            r2_cleaned = float(
                r2_score(targets.iloc[keep_indices], raw_preds[keep_indices])
            )

        return mae, mse, r2, r2_cleaned, preds

    def evaluate_val_performance(
        self,
        data_splits: DataSplits,
        id: str,
        features_to_fit_set: set[FeatureMetadata],
        mem_used: float,
        mem_peak: float,
        use_combined_data: bool,
        params: Optional[T_Params] = None,
        score_cv: Optional[float] = None,
        filter_outliers: bool = False,
        outlier_count: int = prefs.default_worst_errors_n,
    ) -> ModelConfiguration:
        """
        Evaluate model performance and construct a comprehensive ModelConfiguration.

        This method orchestrates task-specific scoring (Classification vs. Regression),
        feature importance extraction, and distribution analysis. It populates a
        standardized configuration object used by the Auditor for model selection.

        Parameters
        ----------
        data_splits : DataSplits
            The container for training and validation datasets.
        id : str
            A unique identifier for the resulting configuration.
        features_to_fit_set : set[FeatureMetadata]
            Metadata defining the specific columns used for evaluation.
        mem_used : float
            Resident memory delta measured during the fit process (bytes).
        mem_peak : float
            Peak Resident Set Size (RSS) measured after fitting (bytes).
        use_combined_data : bool
            Indicator if the model was trained on merged train/validation sets.
        params : T_Params, optional
            Hyperparameters to attach to the config. Defaults to current model_dials.
        score_cv : float, optional
            The cross-validation score from the tuning phase.
        filter_outliers : bool, default False
            If True, calculates 'cleaned' metrics by excluding highest-error samples.
        outlier_count : int, default prefs.default_worst_errors_n
            The number of high-error observations to exclude if filtering.

        Returns
        -------
        ModelConfiguration
            A fully populated container holding metrics, predictions, and stats.
        """
        # Late import to prevent circular dependency at module level
        from dsr_feature_eng_ml.evaluation.schema import ModelConfiguration

        # 1. Setup Context
        active_params = params if params is not None else self.model_dials
        feature_list = [f.name for f in features_to_fit_set]

        eval_features = data_splits.val_features[feature_list]
        eval_target = data_splits.val_target
        train_features = data_splits.train_features[feature_list]
        train_target = data_splits.train_target

        # 2. Task-Specific Scoring
        metrics = self._get_validation_metrics(
            train_features=train_features,
            train_target=train_target,
            eval_features=eval_features,
            eval_target=eval_target,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
        )

        # 3. Feature Importance and Initialization
        importance_analysis = self.analyze_feature_importance(features_to_fit_set)

        config = ModelConfiguration(
            id=id,
            model_type=self.model_type,
            task_type=self.task_type,
            balancing_strategy=self.balancing_strategy,
            optimization_strategy=self.optimization_strategy,
            model_params=active_params,
            cv=self.cv if self.cv is not None else 0,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_iter=self.n_iter,
            max_iter=self.max_iter,
            has_val_set_evaluation_scores=True,
            use_combined_data=use_combined_data,
            score_cv=score_cv,
            score_train=metrics["score_train"],
            score_val=metrics["score_val"],
            score_val_cleaned=metrics["score_val_cleaned"],
            mae_train=metrics["mae_train"],
            mae_val=metrics["mae_val"],
            mse_train=metrics["mse_train"],
            mse_val=metrics["mse_val"],
            r2_train=metrics["r2_train"],
            r2_val=metrics["r2_val"],
            r2_val_cleaned=metrics["r2_val_cleaned"],
            accuracy_train=metrics["accuracy_train"],
            accuracy_val=metrics["accuracy_val"],
            accuracy_val_cleaned=metrics["accuracy_val_cleaned"],
            preds_val=metrics["preds_val"],
            probs_val=metrics["probs_val"],
            acceptable_gap=self.acceptable_gap,
            large_gap=self.large_gap,
            feature_analysis=importance_analysis,
            used_gb=mem_used,
            actual_peak_gb=mem_peak,
            num_candidates=self.num_candidates,
        )

        # 4. Statistical Post-Processing
        from dsr_feature_eng_ml.evaluation import ModelConfigurationStats

        stats = ModelConfigurationStats.from_config(
            data_splits=data_splits, config=config
        )
        train_s = stats.model_split_stats["train"]
        val_s = stats.model_split_stats["val"]

        # Use replace for the final statistical augmentation
        return dataclasses.replace(
            config,
            train_mean=train_s.mean,
            train_std=train_s.std,
            train_median=train_s.median,
            train_skew=train_s.skew,
            train_kurtosis=train_s.kurtosis,
            val_mean=val_s.mean,
            val_std=val_s.std,
            val_median=val_s.median,
            val_skew=val_s.skew,
            val_kurtosis=val_s.kurtosis,
            quality_score=stats.quality_score,
            drift_index=stats.drift_index,
            mean_delta=stats.mean_delta,
            std_delta=stats.std_delta,
        )

    def evaluate_test_set_performance(
        self,
        data_splits: DataSplits,
        config: ModelConfiguration,
        features_to_fit_set: set[FeatureMetadata],
    ) -> ModelConfiguration:
        """
        Evaluate performance on the hold-out test set and update the configuration.

        This method retrains the model on the combined training and validation sets
        using the best found hyperparameters, generates predictions for the test set,
        and calculates final generalization metrics and statistical distributions.

        Parameters
        ----------
        data_splits : DataSplits
            The dataset container providing train, validation, and test splits.
        config : ModelConfiguration
            The existing configuration object (holding validation scores) to be
            augmented with test-set results.
        features_to_fit_set : set[FeatureMetadata]
            The specific set of features used for training and evaluation.

        Returns
        -------
        ModelConfiguration
            An updated copy of the configuration containing test scores, predictions,
            and distribution statistics.
        """
        from dsr_feature_eng_ml.evaluation import ModelConfigurationStats, SplitType

        # 1. Action: Retrain on combined (train + val) data for final test
        _, _ = self.fit(
            data_splits=data_splits,
            features_to_fit_set=features_to_fit_set,
            use_combined_data=True,
        )

        # 2. Setup Context
        feature_list = [f.name for f in features_to_fit_set]
        eval_features = data_splits.test_features[feature_list]
        eval_target = data_splits.test_target

        # 3. Task-Specific Scoring
        test_metrics = self._get_test_metrics(
            eval_features=eval_features,
            eval_target=eval_target,
            filter_outliers=config.filter_outliers,
            outlier_count=config.outlier_count,
        )

        # 4. Update Configuration Metrics
        config = dataclasses.replace(
            config,
            has_test_set_evaluation_scores=True,
            score_test=test_metrics["score_test"],
            mae_test=test_metrics["mae_test"],
            mse_test=test_metrics["mse_test"],
            r2_test=test_metrics["r2_test"],
            accuracy_test=test_metrics["accuracy_test"],
            preds_test=test_metrics["preds_test"],
            probs_test=test_metrics["probs_test"],
        )

        # 5. Statistical Distribution Analysis (Test Set)
        stats = ModelConfigurationStats.from_config(
            data_splits=data_splits, config=config, split_type=SplitType.TEST
        )
        test_s = stats.model_split_stats["test"]

        # Final augmentation with statistical properties
        return dataclasses.replace(
            config,
            test_mean=test_s.mean,
            test_std=test_s.std,
            test_median=test_s.median,
            test_skew=test_s.skew,
            test_kurtosis=test_s.kurtosis,
        )

    @staticmethod
    def find_optimal_threshold(
        data_splits: DataSplits,
        model: ProbabilisticClassifier,
    ) -> tuple[float, float, np.ndarray]:
        """
        Find the classification threshold that maximizes the F1-score.

        This utility trains the provided model, evaluates class probabilities on the
        validation set to locate the optimal threshold via a Precision-Recall curve,
        and then applies that threshold to the test set to determine final performance.

        Parameters
        ----------
        data_splits : DataSplits
            The container for training, validation, and test data splits.
        model : ProbabilisticClassifier
            An uninitialized or pre-configured classifier that supports the
            `predict_proba` method (e.g., RandomForestClassifier, LogisticRegression).

        Returns
        -------
        optimal_threshold : float
            The probability threshold that yielded the highest validation F1-score.
        test_f1_score : float
            The weighted F1-score achieved on the test set using the optimal threshold.
        test_probabilities : np.ndarray
            The raw class-1 probabilities generated for the test set.

        Raises
        ------
        TypeError
            If the provided model does not satisfy the `ProbabilisticClassifier` protocol.
        """
        from sklearn.metrics import f1_score, precision_recall_curve

        # 1. Verify Protocol Compliance
        if not isinstance(model, ProbabilisticClassifier):
            raise TypeError(
                f"The provided model of type {type(model).__name__} does not "
                "support the required 'predict_proba' method for threshold optimization."
            )

        # 2. Train and Predict Probabilities
        model.fit(data_splits.train_features, data_splits.train_target)

        # We target the positive class (index 1) for the threshold search
        valid_proba_array = model.predict_proba(data_splits.val_features)
        valid_proba = valid_proba_array[:, 1]

        # 3. Locate Optimal Threshold on Validation Set
        precisions, recalls, thresholds = precision_recall_curve(
            data_splits.val_target, valid_proba
        )

        # Calculate F1: 2 * (precision * recall) / (precision + recall)
        # We use nan_to_num to handle divisions by zero gracefully
        numerator = 2 * (precisions * recalls)
        denominator = precisions + recalls
        f1_scores = np.nan_to_num(numerator / denominator)

        # Find the index of the maximum F1. Note: thresholds is length n-1
        # relative to precisions/recalls, but the last f1 is usually 0 (recall=0).
        opt_idx = int(np.argmax(f1_scores))

        # Ensure we don't exceed the thresholds array bounds
        actual_idx = min(opt_idx, len(thresholds) - 1)
        optimal_threshold = float(thresholds[actual_idx])

        # 4. Evaluate on Test Set
        test_proba_array = model.predict_proba(data_splits.test_features)
        test_proba = test_proba_array[:, 1]

        # Apply the learned threshold
        y_pred_test = (test_proba >= optimal_threshold).astype(int)
        final_f1_result = float(
            f1_score(data_splits.test_target, y_pred_test, average="weighted")
        )

        return optimal_threshold, final_f1_result, test_proba

    def analyze_feature_importance(
        self, features_to_fit_set: set[FeatureMetadata]
    ) -> ModelFeatureImportance:
        """
        Extract and map feature importance weights to their respective metadata.

        This method retrieves the raw importance values (or absolute coefficients)
        from the fitted estimator and encapsulates them within a structured
        ModelFeatureImportance container for downstream auditing and visualization.

        Parameters
        ----------
        features_to_fit_set : set[FeatureMetadata]
            The set of feature metadata objects corresponding to the columns
            used during the model's training phase.

        Returns
        -------
        ModelFeatureImportance
            A structured container mapping feature names to their importance values.
            Returns an empty instance if the underlying estimator does not
            expose importance attributes.
        """
        # Late import to avoid circular dependencies with schema definitions
        from dsr_feature_eng_ml.evaluation.schema import ModelFeatureImportance

        # The feature_importances property handles the logic for extracting
        # both tree-based importance and linear coefficients.
        importances = self.feature_importances

        if importances is None:
            return ModelFeatureImportance.empty()

        return ModelFeatureImportance(
            feature_set=features_to_fit_set, importances=importances
        )

    def _get_validation_metrics(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series[Any],
        eval_features: pd.DataFrame,
        eval_target: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> dict[str, Any]:
        """Return train/validation metrics in a normalized dictionary shape."""
        if self.task_type == TaskType.CLASSIFICATION:
            acc_train, _, _, _ = self._score_classification(
                features=train_features,
                targets=train_target,
                filter_outliers=filter_outliers,
                outlier_count=outlier_count,
            )
            acc_val, acc_val_cleaned, preds_val, probs_val = self._score_classification(
                features=eval_features,
                targets=eval_target,
                filter_outliers=filter_outliers,
                outlier_count=outlier_count,
            )
            return {
                "score_train": acc_train,
                "score_val": acc_val,
                "score_val_cleaned": acc_val_cleaned,
                "mae_train": None,
                "mae_val": None,
                "mse_train": None,
                "mse_val": None,
                "r2_train": None,
                "r2_val": None,
                "r2_val_cleaned": None,
                "accuracy_train": acc_train,
                "accuracy_val": acc_val,
                "accuracy_val_cleaned": acc_val_cleaned,
                "preds_val": preds_val,
                "probs_val": probs_val,
            }

        mae_train, mse_train, r2_train, _, _ = self._score_regression(
            features=train_features,
            targets=train_target,
            filter_outliers=False,
            outlier_count=0,
        )
        mae_val, mse_val, r2_val, r2_val_cleaned, preds_val = self._score_regression(
            features=eval_features,
            targets=eval_target,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
        )
        return {
            "score_train": r2_train,
            "score_val": r2_val,
            "score_val_cleaned": r2_val_cleaned,
            "mae_train": mae_train,
            "mae_val": mae_val,
            "mse_train": mse_train,
            "mse_val": mse_val,
            "r2_train": r2_train,
            "r2_val": r2_val,
            "r2_val_cleaned": r2_val_cleaned,
            "accuracy_train": None,
            "accuracy_val": None,
            "accuracy_val_cleaned": None,
            "preds_val": preds_val,
            "probs_val": None,
        }

    def _get_test_metrics(
        self,
        eval_features: pd.DataFrame,
        eval_target: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> dict[str, Any]:
        """Return hold-out test metrics in a normalized dictionary shape."""
        if self.task_type == TaskType.CLASSIFICATION:
            acc_test, _, preds_test, probs_test = self._score_classification(
                features=eval_features,
                targets=eval_target,
                filter_outliers=filter_outliers,
                outlier_count=outlier_count,
            )
            return {
                "score_test": acc_test,
                "mae_test": None,
                "mse_test": None,
                "r2_test": None,
                "accuracy_test": acc_test,
                "preds_test": preds_test,
                "probs_test": probs_test,
            }

        mae_test, mse_test, r2_test, _, preds_test = self._score_regression(
            features=eval_features,
            targets=eval_target,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
        )
        return {
            "score_test": r2_test,
            "mae_test": mae_test,
            "mse_test": mse_test,
            "r2_test": r2_test,
            "accuracy_test": None,
            "preds_test": preds_test,
            "probs_test": None,
        }

    @staticmethod
    def instantiate_model(
        model_cls: type[ModelSpecification],
        strategy: BalancingStrategy,
        params: Optional[ModelParams],
        cv: int,
        optimization_strategy: OptimizationStrategy,
        **kwargs: Any,
    ) -> ModelSpecification:
        """Factory method to instantiate a concrete ModelSpecification."""
        import inspect

        all_kwargs = {
            "balancing_strategy": strategy,
            "params": params,
            "cv": cv,
            "optimization_strategy": optimization_strategy,
            **kwargs,
        }
        # Filter to only the parameters the target class actually accepts so
        # that a stored field like max_iter doesn't break tree-based models
        # whose __init__ signatures don't include it.
        accepted = set(inspect.signature(model_cls.__init__).parameters)
        init_kwargs = {k: v for k, v in all_kwargs.items() if k in accepted}
        return model_cls(**init_kwargs)

    @classmethod
    def create_model_from_config(
        cls, config: ModelConfiguration
    ) -> Optional[ModelSpecification]:
        """Reconstruct a ModelSpecification instance from a configuration object."""
        model_cls = config.model_type.model_class

        if model_cls is not None:
            return cls.instantiate_model(
                model_cls=model_cls,
                strategy=config.balancing_strategy,
                params=config.model_params,
                cv=config.cv,
                optimization_strategy=config.optimization_strategy,
                scoring=config.scoring,
                n_jobs=config.n_jobs,
                n_iter=config.n_iter,
                max_iter=config.max_iter,
                acceptable_gap=config.acceptable_gap,
                large_gap=config.large_gap,
            )

        print(
            f"⚠️ Unable to instantiate model class: ModelType.{config.model_type.name} has no associated class."
        )
        return None


class ClassificationModelSpecification(ModelSpecification[T_Params, T_Estimator], ABC):
    """Task-specialized base class for classification model specifications."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION

    def get_valid_scoring_metrics(self) -> list[ScoringMetric]:
        return ScoringMetric.get_valid_metrics(TaskType.CLASSIFICATION)

    def resolve_search_scoring_key(self) -> str:
        return {
            ScoringMetric.F1: "f1_weighted",
            ScoringMetric.PRECISION: "precision_weighted",
            ScoringMetric.RECALL: "recall_weighted",
        }.get(self.scoring, self.scoring.value)

    def _get_validation_metrics(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series[Any],
        eval_features: pd.DataFrame,
        eval_target: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> dict[str, Any]:
        acc_train, _, _, _ = self._score_classification(
            features=train_features,
            targets=train_target,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
        )
        acc_val, acc_val_cleaned, preds_val, probs_val = self._score_classification(
            features=eval_features,
            targets=eval_target,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
        )
        return {
            "score_train": acc_train,
            "score_val": acc_val,
            "score_val_cleaned": acc_val_cleaned,
            "mae_train": None,
            "mae_val": None,
            "mse_train": None,
            "mse_val": None,
            "r2_train": None,
            "r2_val": None,
            "r2_val_cleaned": None,
            "accuracy_train": acc_train,
            "accuracy_val": acc_val,
            "accuracy_val_cleaned": acc_val_cleaned,
            "preds_val": preds_val,
            "probs_val": probs_val,
        }

    def _get_test_metrics(
        self,
        eval_features: pd.DataFrame,
        eval_target: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> dict[str, Any]:
        acc_test, _, preds_test, probs_test = self._score_classification(
            features=eval_features,
            targets=eval_target,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
        )
        return {
            "score_test": acc_test,
            "mae_test": None,
            "mse_test": None,
            "r2_test": None,
            "accuracy_test": acc_test,
            "preds_test": preds_test,
            "probs_test": probs_test,
        }


class RegressionModelSpecification(ModelSpecification[T_Params, T_Estimator], ABC):
    """Task-specialized base class for regression model specifications."""

    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION

    def get_valid_scoring_metrics(self) -> list[ScoringMetric]:
        return ScoringMetric.get_valid_metrics(TaskType.REGRESSION)

    def _get_validation_metrics(
        self,
        train_features: pd.DataFrame,
        train_target: pd.Series[Any],
        eval_features: pd.DataFrame,
        eval_target: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> dict[str, Any]:
        mae_train, mse_train, r2_train, _, _ = self._score_regression(
            features=train_features,
            targets=train_target,
            filter_outliers=False,
            outlier_count=0,
        )
        mae_val, mse_val, r2_val, r2_val_cleaned, preds_val = self._score_regression(
            features=eval_features,
            targets=eval_target,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
        )
        return {
            "score_train": r2_train,
            "score_val": r2_val,
            "score_val_cleaned": r2_val_cleaned,
            "mae_train": mae_train,
            "mae_val": mae_val,
            "mse_train": mse_train,
            "mse_val": mse_val,
            "r2_train": r2_train,
            "r2_val": r2_val,
            "r2_val_cleaned": r2_val_cleaned,
            "accuracy_train": None,
            "accuracy_val": None,
            "accuracy_val_cleaned": None,
            "preds_val": preds_val,
            "probs_val": None,
        }

    def _get_test_metrics(
        self,
        eval_features: pd.DataFrame,
        eval_target: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> dict[str, Any]:
        mae_test, mse_test, r2_test, _, preds_test = self._score_regression(
            features=eval_features,
            targets=eval_target,
            filter_outliers=filter_outliers,
            outlier_count=outlier_count,
        )
        return {
            "score_test": r2_test,
            "mae_test": mae_test,
            "mse_test": mse_test,
            "r2_test": r2_test,
            "accuracy_test": None,
            "preds_test": preds_test,
            "probs_test": None,
        }
