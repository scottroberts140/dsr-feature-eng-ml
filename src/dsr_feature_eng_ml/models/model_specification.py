"""Base model specification and shared training utilities."""

from __future__ import annotations
from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import psutil
import os
from typing import (
    Union,
    Optional,
    cast,
    Mapping,
    Type,
    Any,
    Final,
    TypeVar,
    Protocol,
    Generic,
    runtime_checkable,
    Callable,
    Tuple,
    TYPE_CHECKING,
    Literal,
    Self,
    TypeGuard,
)
from dsr_utils.formatting import (
    BoolRepresentation,
    PercentageFormat,
    IntegerFormat,
    EnumFormat,
    BoolFormat,
)

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from dsr_feature_eng_ml.enums import (
    ModelType,
    BalancingStrategy,
    OptimizationStrategy,
    TaskType,
    ScoringMetric,
)

if TYPE_CHECKING:
    from dsr_feature_eng_ml.evaluation.schema import (
        DataSplits,
        ModelConfiguration,
        ModelFeatureImportance,
        FeatureMetadata,
    )

from dsr_feature_eng_ml.preferences import prefs
from dsr_feature_eng_ml.utils.generalization import calculate_generalization_status

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_curve,
)


@runtime_checkable
class ScikitModel(Protocol):
    """Protocol for scikit-learn compatible estimators.

    Defines the minimal interface required for a scikit-learn estimator:
    fit() for training and predict() for inference. Used for type hints
    when working with generic sklearn models.

    Methods:
        fit: Train the model on data (X, y).
        predict: Generate predictions from input X.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Self: ...
    def predict(self, X: Any) -> Any: ...
    def get_params(self, deep: bool = True) -> Mapping[str, Any]: ...


@runtime_checkable
class ProbabilisticClassifier(ScikitModel, Protocol):
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        ...


T_Params = TypeVar("T_Params", bound="ModelParams")
T_Estimator = TypeVar("T_Estimator", bound="ScikitModel")


@dataclass(frozen=True)
class ModelParams(ABC):
    """Base dataclass for model-specific hyperparameters.

    Acts as a lightweight, immutable container that model parameter classes
    can inherit from. Provides a standardized ``to_dict()`` for converting
    dataclass fields to a parameter mapping suitable for sklearn estimators,
    and requires subclasses to implement ``info()`` for human-readable summaries.

    Attributes:
        random_state: Optional seed used by estimators for reproducibility.
            Defaults to ``None`` (no fixed seed).
    """

    random_state: Optional[int] = None
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL

    @abstractmethod
    def info(self) -> str:
        pass

    def to_dict(self) -> dict:
        """Standardizes parameter mapping and handles Enum serialization."""
        import enum

        data = {}
        for k, v in dataclasses.asdict(self).items():
            if v is None:
                continue

            # If the value is an Enum, extract the value (e.g., 'MANUAL')
            # so JSON can handle it.
            if isinstance(v, enum.Enum):
                data[k] = v.name
            else:
                data[k] = v

        return data

    @property
    def num_candidates(self) -> int:
        if self.optimization_strategy == OptimizationStrategy.MANUAL:
            return 1

        if self.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            return ModelSpecification.calculate_optimal_n_iter(
                self.to_dict(), min_iter=-1
            )

        params_dict = self.to_dict()

        # Multiply lengths of all list-based parameters (Cartesian Product)
        import math

        nc = math.prod(
            [len(v) if isinstance(v, list) else 1 for v in params_dict.values()]
        )
        return nc

    @staticmethod
    @abstractmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list]:
        """Every subclass must implement this."""
        pass


class ModelSpecification(ABC, Generic[T_Params, T_Estimator]):
    """Abstract base class for model specifications with common training parameters.

    Provides shared functionality for model training, validation prediction, and
    performance evaluation. Cannot be instantiated directly - must be subclassed
    by specific model types (Decision Tree, Random Forest, Logistic Regression).

    This class uses managed mutability: the predicted_train and predicted_val
    attributes are modified by the calc_predictions() method during model evaluation,
    but this is controlled through the class interface rather than external manipulation.

    Attributes:
        estimator_class (Callable[..., T_Estimator]): Sklearn estimator class reference (e.g., DecisionTreeClassifier).
        params_class (Type[T_Params]): ModelParams subclass for hyperparameter configuration.
        cv (int): Number of cross-validation folds.
        balancing_strategy (BalancingStrategy): Strategy for class balancing.
        scoring (ScoringMetric): Scoring metric for model evaluation.
        n_jobs (int): Number of parallel jobs (-1 for all CPUs).
        n_iter (int): Number of iterations for randomized search.
        max_iter (int): Max iterations for estimators that support it.
        task_type (TaskType): Type of ML task (default: TaskType.CLASSIFICATION).
        model_dials (T_Params): Model-specific hyperparameter container.
        predicted_train (pd.Series): Training set predictions (modified during evaluation).
        predicted_val (pd.Series): Validation set predictions (modified during evaluation).
        optimization_strategy (OptimizationStrategy): Manual vs randomized search.

    Example:
        >>> # Cannot instantiate directly - use subclasses
        >>> dtree = DecisionTree(
        ...     cv=5,
        ...     balancing_strategy=BalancingStrategy.NONE,
        ...     task_type=TaskType.CLASSIFICATION,
        ... )

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses inherit shared functionality while implementing model-specific behavior.
    """

    params_class: Type[T_Params]

    @abstractmethod
    def get_estimator_class(self) -> Type[T_Estimator]:
        """Return the sklearn estimator class for this model."""
        pass

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Return the model task type (classification or regression)."""
        pass

    @property
    @abstractmethod
    def scoring(self) -> ScoringMetric:
        """Return the scoring metric used for evaluation."""
        pass

    @scoring.setter
    @abstractmethod
    def scoring(self, value: ScoringMetric):
        """Set the scoring metric used for evaluation."""
        pass

    @property
    def num_candidates(self) -> int:
        """Return the number of hyperparameter candidates to evaluate."""
        return self.model_dials.num_candidates

    @property
    def total_fits(self) -> int:
        """Return the total number of estimator fits for the search."""
        from dsr_utils.formatting import format_label_value_pairs

        cv_count = self.cv if self.cv is not None else 5

        if self.optimization_strategy == OptimizationStrategy.MANUAL:
            return cv_count

        nc = self.num_candidates

        if nc is None:
            nc = self.n_iter if self.n_iter != -1 else 1

        return cv_count * nc

    def __init__(
        self,
        cv: Optional[int],
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: Optional[Any] = None,
        n_jobs: int = 3,
        n_iter: int = -1,
        max_iter: int = 1000,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize shared model configuration parameters.

        Args:
            cv: Number of CV folds to use.
            balancing_strategy: Resampling/weighting strategy.
            params: Optional model parameter object.
            n_jobs: Parallel job count.
            n_iter: Iterations for randomized search (-1 for auto).
            max_iter: Max estimator iterations where supported.
            acceptable_gap: Gap threshold for marginal generalization.
            large_gap: Gap threshold for overfit generalization.
            optimization_strategy: Manual, grid, or randomized search.
        """
        self.cv = cv
        self.balancing_strategy = balancing_strategy
        valid_metrics = ScoringMetric.get_valid_metrics(self.task_type)

        if self.scoring not in valid_metrics:
            valid_names = [m.name for m in valid_metrics]
            enum_format = EnumFormat(use_value=False)

            raise ValueError(
                f"Invalid scoring metric '{enum_format.format_value(self.scoring)}' for task type {enum_format.format_value(self.task_type)}. "
                f"Valid options are: {valid_names}"
            )

        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.acceptable_gap = acceptable_gap
        self.large_gap = large_gap
        self.predicted_train = pd.Series(dtype=float)
        self.predicted_val = pd.Series(dtype=float)
        self.optimization_strategy = optimization_strategy

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Every model must identify itself."""
        pass

    @property
    @abstractmethod
    def model_dials(self) -> T_Params:
        """Every model must return its parameters."""
        pass

    @model_dials.setter
    @abstractmethod
    def model_dials(self, value: T_Params) -> None:
        """Update the current model parameters."""
        pass

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importance values if the estimator exposes them."""
        if self.estimator is None:
            return None

        # Use getattr with a default of None to satisfy Pylance
        # We cast to Optional[np.ndarray] so the rest of our code knows the type

        # 1. Try Tree-based
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is not None:
            return cast(np.ndarray, importances)

        # 2. Try Linear-based
        coef = getattr(self.estimator, "coef_", None)
        if coef is not None:
            raw_coef = cast(np.ndarray, coef)
            return np.abs(raw_coef).flatten()

        return None

    @abstractmethod
    def create_estimator(self, parameters: Optional[T_Params] = None) -> T_Estimator:
        """Create a configured estimator instance from parameters."""
        pass

    def is_probabilistic(self, estimator: Any) -> TypeGuard[ProbabilisticClassifier]:
        """Runtime and static check for probability support."""
        return hasattr(estimator, "predict_proba")

    @classmethod
    def calculate_optimal_n_iter(
        cls, grid: dict, min_iter: int = -1, max_iter: int = 100, coverage: float = 0.10
    ) -> int:
        """
        Calculates n_iter based on a percentage of the total search space.
        """
        import math

        # 1. Calculate the total size of the grid
        total_combinations = 1
        for values in grid.values():
            # If it's a distribution, it has infinite size.
            # We treat it as a "large" space to trigger max_iter.
            if hasattr(values, "rvs"):
                total_combinations = int(
                    max_iter / coverage
                )  # Force the math to hit max_iter
                break

            if isinstance(values, list):
                total_combinations *= len(values)

        # 2. Apply coverage (e.g., try to hit 10% of the space)
        calculated_iter = int(math.ceil(total_combinations * coverage))

        # If the grid is very small (e.g., < 10 combinations), searching 10% is useless.
        # Use max(calculated_iter, min(total_combinations, 10)) or a fixed floor.
        actual_iter = max(calculated_iter, min(total_combinations, 5))

        # 3. Clip the results to stay within reasonable time limits
        return max(min_iter, min(actual_iter, max_iter))

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
        """Run hyperparameter search and update model parameters.

        Args:
            data_splits: Train/validation splits used for tuning.
            method: Optimization strategy (grid or random search).
            features_to_fit_set: Feature metadata set used to select columns.
            custom_grid: Optional search grid override.
            use_combined_data: If True, tune on train+val combined.
            max_sample_size: Optional cap on rows for faster tuning.
            perform_memory_check: If True, estimate memory risk before tuning.

        Returns:
            Tuple of (best_params, best_score, risk_triggered, available_gb,
            estimated_peak_gb, model_multiplier, sampling_factor).
        """
        if use_combined_data:
            # Combine the already-scaled features and targets
            features = pd.concat([data_splits.train_features, data_splits.val_features])
            target = pd.concat([data_splits.train_target, data_splits.val_target])
        else:
            features = data_splits.train_features
            target = data_splits.train_target

        feature_list = [f.name for f in features_to_fit_set]
        features = features[feature_list]
        total_rows = len(features)
        memory_risk_triggered: bool = False
        estimated_peak_gb: float = 0.0
        available_gb: float = 0.0
        model_multiplier: float = 1.0
        sampling_factor: float = 1.0

        def get_tuning_samples(
            features: pd.DataFrame, target: pd.Series[Any], rows: int, random_state: int
        ) -> Tuple[pd.DataFrame, pd.Series[Any]]:
            tuning_features = features.sample(n=rows, random_state=random_state)
            tuning_target = target.loc[tuning_features.index]
            return tuning_features, tuning_target

        if max_sample_size is None:
            # Heuristic: Cap at 10% of data OR 500k rows, whichever is smaller
            # to ensure the "Tune" phase remains fast (under 2 mins)
            max_sample_size = min(prefs.min_target_tuning_rows, int(total_rows * 0.10))

        sample_size_format = IntegerFormat()
        sampling_factor_format = PercentageFormat(precision=1)

        if total_rows > max_sample_size:
            sampling_factor = max_sample_size / total_rows
            total_rows_format = IntegerFormat()
            print(
                f"⚠️ Dataset size ({total_rows_format.format_value(total_rows)}) exceeds tuning safety limit."
            )
            print(
                f"📉 Sampling {sample_size_format.format_value(max_sample_size)} rows ({sampling_factor_format.format_value(sampling_factor)}) for optimization phase..."
            )

            tuning_features, tuning_target = get_tuning_samples(
                features, target, max_sample_size, data_splits.random_state
            )
        else:
            tuning_features = features
            tuning_target = target

        if perform_memory_check:
            from dsr_feature_eng_ml.utils.memory import check_memory_risk

            memory_risk_triggered, estimated_peak_gb, available_gb, model_multiplier = (
                check_memory_risk(tuning_features, self, self.n_jobs)
            )
            risk_format = BoolFormat(representation=BoolRepresentation.YES_NO)
            print(
                f"Risk: {risk_format.format_value(memory_risk_triggered)} | Estimated peak: {prefs.gb_format.format_value(estimated_peak_gb)} | Available: {prefs.gb_format.format_value(available_gb)}"
            )

        if memory_risk_triggered:
            print(
                f"🚨 DANGER: Predicted peak memory {prefs.gb_format.format_value(estimated_peak_gb)} exceeds available {prefs.gb_format.format_value(available_gb)}."
            )
            max_sample_size = int(
                int(total_rows * (available_gb / estimated_peak_gb) * 0.7)
            )
            sampling_factor = max_sample_size / total_rows
            tuning_features, tuning_target = get_tuning_samples(
                features, target, max_sample_size, data_splits.random_state
            )
            print(
                f"📉 Sampling {sample_size_format.format_value(max_sample_size)} rows ({sampling_factor_format.format_value(sampling_factor)}) for optimization phase..."
            )

        # 1. Get the grid: Either passed in, or the standard one for this model type
        if custom_grid is None:
            # Check if the existing params object already contains lists (a search space)
            potential_grid = dataclasses.asdict(self.model_dials)
            search_space = {
                k: v for k, v in potential_grid.items() if isinstance(v, (list, tuple))
            }

            if search_space:
                grid = search_space
            else:
                grid = self.params_class.get_standard_search_grid(narrow=True)
        else:
            grid = custom_grid

        # 2. Detect the REAL strategy
        # If any value is a distribution (has .rvs), we are forced into Random Search
        is_dist_search = any(hasattr(v, "rvs") for v in grid.values())

        if is_dist_search or method == OptimizationStrategy.RANDOM_SEARCH:
            self.optimization_strategy = OptimizationStrategy.RANDOM_SEARCH
        else:
            self.optimization_strategy = OptimizationStrategy.GRID_SEARCH

        # 3. Execution
        search_cv_model = cast(BaseEstimator, self.create_estimator())

        def _prepare_search_grid(estimator, custom_grid: dict) -> dict:
            """
            1. Filters out keys that aren't valid for the estimator.
            2. Wraps scalars in lists to satisfy Scikit-Learn SearchCV requirements.
            """
            from collections.abc import Iterable

            # Get the valid parameters for this specific Scikit-Learn model
            valid_params = estimator.get_params().keys()

            clean_grid = {}
            for k, v in custom_grid.items():
                if k in valid_params:
                    # Check if it's already a list/dist (Scikit-Learn requirement)
                    if isinstance(v, (list, tuple)) or hasattr(v, "rvs"):
                        clean_grid[k] = v
                    else:
                        # Wrap scalar (75 -> [75])
                        clean_grid[k] = [v]

            return clean_grid

        if self.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
            refined_grid = _prepare_search_grid(search_cv_model, grid)

            # Dynamically set n_iter if it was passed as -1 or 'auto'
            if self.n_iter == -1:
                self.n_iter = self.calculate_optimal_n_iter(refined_grid)

            search_cv = RandomizedSearchCV(
                estimator=search_cv_model,
                param_distributions=refined_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=self.scoring.value,
                n_jobs=self.n_jobs,
                verbose=prefs.cv_verbose,
                random_state=data_splits.random_state,  # Use data seed for search
            )
        else:
            search_cv = GridSearchCV(
                estimator=search_cv_model,
                param_grid=grid,
                cv=self.cv,
                scoring=self.scoring.value,
                n_jobs=self.n_jobs,
                verbose=prefs.cv_verbose,
            )

        # 4. Fit using data_splits attributes
        search_cv.fit(tuning_features, tuning_target)

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
        """Fit the estimator and return memory usage stats.

        Args:
            data_splits: Train/validation splits used for fitting.
            features_to_fit_set: Feature metadata used to select input columns.
            use_combined_data: If True, fit on train+val combined data.

        Returns:
            Tuple of (memory_used_bytes, peak_rss_bytes).
        """
        # 1. Get the data (Resampled if OVERSAMPLED/UNDERSAMPLED, otherwise original)
        X, y = data_splits.get_balanced_train_data(
            strategy=self.balancing_strategy,
            feature_set=features_to_fit_set,
            use_combined_data=use_combined_data,
        )

        # 2. Get the weights (Only non-None if strategy is WEIGHTED)
        # Check task_type property to decide if it's regression
        weights = data_splits.get_train_weights(
            self.balancing_strategy,
            is_regression=(self.task_type == TaskType.REGRESSION),
        )

        # 3. Fit the estimator
        # Most scikit-learn estimators accept sample_weight=None by default
        self.estimator = self.create_estimator()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        self.estimator.fit(X, y, sample_weight=weights)
        mem_after = process.memory_info().rss
        mem_used = mem_after - mem_before
        return mem_used, mem_after

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
        """Fit the model and return a validation-scored configuration.

        Args:
            data_splits: Train/validation splits.
            id: Identifier for the resulting configuration.
            features_to_fit_set: Feature metadata used to select input columns.
            score_cv: Optional CV score to include in the config.
            use_combined_data: If True, fit on train+val combined data.
            filter_outliers: If True, remove worst errors for cleaned metrics.
            outlier_count: Number of worst errors to treat as outliers.

        Returns:
            A `ModelConfiguration` populated with validation metrics and stats.
        """
        # 1. Action: Fit the model
        mem_used, mem_peak = self.fit(
            data_splits=data_splits,
            features_to_fit_set=features_to_fit_set,
            use_combined_data=use_combined_data,
        )

        # 2. Analysis: Generate metrics and return the result
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
        """Compute classification scores and prediction outputs.

        Args:
            features: Feature matrix for scoring.
            targets: Ground-truth labels.
            filter_outliers: If True, drop the most confident incorrect predictions.
            outlier_count: Number of outliers to remove when filtering.

        Returns:
            Tuple of (f1, f1_cleaned, preds, probs).
        """
        prob_estimator = cast(ProbabilisticClassifier, self.estimator)

        # 1. Get discrete predictions and probabilities
        raw_preds = prob_estimator.predict(features)
        raw_probs = prob_estimator.predict_proba(features)

        # 2. Wrap in a Series with the correct index
        preds = pd.Series(raw_preds, index=targets.index)
        probs = pd.DataFrame(raw_probs, index=targets.index)

        # 3. Calculate Standard F1
        f1 = float(f1_score(targets, preds, average="weighted"))

        if filter_outliers:
            # Identify outliers: Wrong predictions where confidence was highest
            # Create a mask where prediction != actual
            incorrect_mask = targets.to_numpy() != raw_preds

            # Get the confidence score for the predicted class
            # raw_probs is (N, classes). np.max gives the confidence of the chosen class.
            confidences = np.max(raw_probs, axis=1)

            # We only care about confidence when the model was WRONG.
            # Set confidence to -1 for correct predictions so they aren't 'top' outliers.
            error_scores = np.where(incorrect_mask, confidences, -1.0)

            # Drop the top N most confident mistakes
            n_to_keep = len(error_scores) - outlier_count
            keep_indices = np.argpartition(error_scores, n_to_keep)[:n_to_keep]

            # Calculate "Cleaned" F1
            f1_cleaned = float(
                f1_score(
                    targets.iloc[keep_indices],
                    raw_preds[keep_indices],
                    average="weighted",
                )
            )
        else:
            f1_cleaned = None

        return f1, f1_cleaned, preds, probs

    def _score_regression(
        self,
        features: pd.DataFrame,
        targets: pd.Series[Any],
        filter_outliers: bool,
        outlier_count: int,
    ) -> tuple[float, float, float, Optional[float], pd.Series]:
        """Compute regression metrics and predictions.

        Args:
            features: Feature matrix for scoring.
            targets: Ground-truth targets.
            filter_outliers: If True, drop largest absolute errors.
            outlier_count: Number of outliers to remove when filtering.

        Returns:
            Tuple of (mae, mse, r2, r2_cleaned, preds).
        """
        # Generate predictions internally to ensure index matching
        raw_preds = self.estimator.predict(features)
        preds = pd.Series(raw_preds, index=targets.index)

        mae = float(mean_absolute_error(targets, preds))
        mse = float(mean_squared_error(targets, preds))
        r2 = float(r2_score(targets, preds))

        if filter_outliers:
            # Identify and remove outliers
            abs_errors = np.abs((targets.to_numpy() - raw_preds).flatten())

            # Get indices of the rows with the smallest errors (dropping the top N)
            # We use partition for efficiency (O(n) vs O(n log n))
            n_to_keep = len(abs_errors) - outlier_count
            keep_indices = np.argpartition(abs_errors, n_to_keep)[:n_to_keep]

            # Calculate "Cleaned" R2
            r2_cleaned = r2_score(targets.iloc[keep_indices], raw_preds[keep_indices])
        else:
            r2_cleaned = None

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
        """Evaluate validation performance and build a ModelConfiguration.

        Computes task-specific metrics and attaches feature importance and
        distribution stats.

        Args:
            data_splits: Train/validation splits used for evaluation.
            id: Identifier for the resulting configuration.
            features_to_fit_set: Feature metadata used to select columns.
            mem_used: Bytes used during fit.
            mem_peak: Peak RSS bytes during fit.
            use_combined_data: If True, model was fit on train+val combined data.
            params: Optional params override (defaults to `model_dials`).
            score_cv: Optional CV score to include.
            filter_outliers: If True, compute cleaned metrics by dropping worst errors.
            outlier_count: Number of outliers to drop.

        Returns:
            Populated `ModelConfiguration` with validation metrics and stats.
        """
        from dsr_feature_eng_ml.evaluation.schema import ModelConfiguration

        # 1. Type Guard: Ensure we have params
        # If None, fall back to the defaults in self.model_dials
        active_params = params if params is not None else self.model_dials
        feature_list = [f.name for f in features_to_fit_set]

        # Select features/target for evaluation
        eval_features = data_splits.val_features
        eval_target = data_splits.val_target
        eval_features = eval_features[feature_list]

        # Handle Task-Specific Scoring
        if self.task_type == TaskType.CLASSIFICATION:
            # Get values
            acc_train, _, _, _ = self._score_classification(
                features=data_splits.train_features[feature_list],
                targets=data_splits.train_target,
                filter_outliers=filter_outliers,
                outlier_count=outlier_count,
            )
            acc_val, acc_val_cleaned, preds_val, probs_val = self._score_classification(
                features=eval_features,
                targets=eval_target,
                filter_outliers=filter_outliers,
                outlier_count=outlier_count,
            )

            # Assign to primary pointers
            score_train = acc_train
            score_val = acc_val
            score_val_cleaned = acc_val_cleaned

            # Assign to specific attributes
            accuracy_train = acc_train
            accuracy_val = acc_val
            accuracy_val_cleaned = acc_val_cleaned

            # Initialize regression metrics as None for classification results
            mae_train = mse_train = r2_train = None
            mae_val = mse_val = r2_val = r2_val_cleaned = None
        else:
            # Regression path: also passing DataFrames now
            mae_train, mse_train, r2_train, _, _ = self._score_regression(
                features=data_splits.train_features[feature_list],
                targets=data_splits.train_target,
                filter_outliers=False,
                outlier_count=0,
            )
            score_train = r2_train

            mae_val, mse_val, r2_val, r2_val_cleaned, preds_val = (
                self._score_regression(
                    features=eval_features,
                    targets=eval_target,
                    filter_outliers=filter_outliers,
                    outlier_count=outlier_count,
                )
            )
            score_val = r2_val
            score_val_cleaned = r2_val_cleaned

            # Initialize classification metrics as None for regression results
            accuracy_train = accuracy_val = accuracy_val_cleaned = probs_val = None

        # 2. Extract Importance Analysis
        importance_analysis = self.analyze_feature_importance(
            features_to_fit_set=features_to_fit_set
        )

        config: ModelConfiguration = ModelConfiguration(
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
            score_train=score_train,
            score_val=score_val,
            score_val_cleaned=score_val_cleaned,
            mae_train=mae_train,
            mae_val=mae_val,
            mse_train=mse_train,
            mse_val=mse_val,
            r2_train=r2_train,
            r2_val=r2_val,
            r2_val_cleaned=r2_val_cleaned,
            accuracy_train=accuracy_train,
            accuracy_val=accuracy_val,
            accuracy_val_cleaned=accuracy_val_cleaned,
            preds_val=preds_val,
            probs_val=probs_val,
            acceptable_gap=self.acceptable_gap,
            large_gap=self.large_gap,
            feature_analysis=importance_analysis,
            used_gb=mem_used,
            actual_peak_gb=mem_peak,
            num_candidates=self.num_candidates,
        )

        from dsr_feature_eng_ml.evaluation import ModelConfigurationStats

        stats = ModelConfigurationStats.from_config(
            data_splits=data_splits, config=config
        )
        train_stats = stats.model_split_stats["train"]
        val_stats = stats.model_split_stats["val"]
        config = dataclasses.replace(
            config,
            train_mean=train_stats.mean,
            train_std=train_stats.std,
            train_median=train_stats.median,
            train_skew=train_stats.skew,
            train_kurtosis=train_stats.kurtosis,
            val_mean=val_stats.mean,
            val_std=val_stats.std,
            val_median=val_stats.median,
            val_skew=val_stats.skew,
            val_kurtosis=val_stats.kurtosis,
            quality_score=stats.quality_score,
            drift_index=stats.drift_index,
            mean_delta=stats.mean_delta,
            std_delta=stats.std_delta,
        )

        return config

    def evaluate_test_set_performance(
        self,
        data_splits: DataSplits,
        config: ModelConfiguration,
        features_to_fit_set: set[FeatureMetadata],
    ) -> ModelConfiguration:
        """Evaluate test-set performance and return an updated configuration.

        Args:
            data_splits: Train/validation/test splits.
            config: Existing configuration to update with test metrics.
            features_to_fit_set: Feature metadata used to select columns.

        Returns:
            Updated `ModelConfiguration` with test metrics and stats.
        """
        from dsr_feature_eng_ml.evaluation import (
            ModelConfigurationStats,
            SplitType,
        )

        # Fit the model
        _, _ = self.fit(
            data_splits=data_splits,
            features_to_fit_set=features_to_fit_set,
            use_combined_data=True,
        )

        # Select features/target for evaluation
        eval_features = data_splits.test_features
        eval_target = data_splits.test_target
        feature_list = [f.name for f in features_to_fit_set]
        eval_features = eval_features[feature_list]

        # Handle Task-Specific Scoring
        if self.task_type == TaskType.CLASSIFICATION:
            # Get values
            acc_test, _, preds_test, probs_test = self._score_classification(
                features=eval_features,
                targets=eval_target,
                filter_outliers=config.filter_outliers,
                outlier_count=config.outlier_count,
            )

            # Assign to primary pointers
            score_test = acc_test

            # Assign to specific attributes
            accuracy_test = acc_test

            # Initialize regression metrics as None for classification results
            mae_test = mse_test = r2_test = None
        else:
            # Regression path: also passing DataFrames
            mae_test, mse_test, r2_test, _, preds_test = self._score_regression(
                features=eval_features,
                targets=eval_target,
                filter_outliers=config.filter_outliers,
                outlier_count=config.outlier_count,
            )
            score_test = r2_test
            accuracy_test = None

            # Initialize classification metrics as None for regression results
            probs_test = None

        config = dataclasses.replace(
            config,
            has_test_set_evaluation_scores=True,
            score_test=score_test,
            mae_test=mae_test,
            mse_test=mse_test,
            r2_test=r2_test,
            accuracy_test=accuracy_test,
            preds_test=preds_test,
            probs_test=probs_test,
        )
        stats = ModelConfigurationStats.from_config(
            data_splits=data_splits, config=config, split_type=SplitType.TEST
        )
        test_stats = stats.model_split_stats["test"]
        config = dataclasses.replace(
            config,
            test_mean=test_stats.mean,
            test_std=test_stats.std,
            test_median=test_stats.median,
            test_skew=test_stats.skew,
            test_kurtosis=test_stats.kurtosis,
        )

        return config

    @staticmethod
    def find_optimal_threshold(
        data_splits: DataSplits,
        model: ProbabilisticClassifier,
    ) -> tuple[float, float, np.ndarray]:
        """Find optimal classification threshold by maximizing F1 score.

        Generic method supporting any sklearn-compatible classifier with predict_proba(),
        including DecisionTreeClassifier, RandomForestClassifier, and LogisticRegression.

        Args:
            data_splits (DataSplits): Training, validation, and test data splits.
            model (ProbabilisticClassifier): Untrained classifier with fit() and predict_proba().

        Returns:
            tuple[float, float, np.ndarray]: (optimal_threshold, test_f1_score, test_probabilities)
        """

        if not isinstance(model, ProbabilisticClassifier):
            raise TypeError(
                f"Model {type(model)} does not support probability predictions."
            )

        model.fit(data_splits.train_features, data_splits.train_target)
        valid_proba_array = model.predict_proba(data_splits.val_features)
        valid_proba = valid_proba_array[:, 1]  # type: ignore[misc]

        precisions, recalls, thresholds = precision_recall_curve(
            data_splits.val_target, valid_proba
        )

        f1_scores = np.nan_to_num(2 * (precisions * recalls) / (precisions + recalls))

        optimal_threshold_index = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_threshold_index]

        test_proba_array = model.predict_proba(data_splits.test_features)
        test_proba = test_proba_array[:, 1]  # type: ignore[misc]
        y_pred_test = (test_proba >= optimal_threshold).astype(int)
        final_f1_result = cast(float, f1_score(data_splits.test_target, y_pred_test))

        return optimal_threshold, final_f1_result, test_proba

    def analyze_feature_importance(
        self, features_to_fit_set: set[FeatureMetadata]
    ) -> ModelFeatureImportance:
        """Create feature-importance analysis from the current estimator.

        Args:
            features_to_fit_set: Feature metadata used to map importances.

        Returns:
            `ModelFeatureImportance` instance (empty if unsupported).
        """
        from dsr_feature_eng_ml.evaluation.schema import ModelFeatureImportance

        importances = self.feature_importances

        if importances is None:
            return ModelFeatureImportance.empty()

        return ModelFeatureImportance(
            feature_set=features_to_fit_set, importances=importances
        )

    @staticmethod
    def instantiate_model(
        model_cls: type[ModelSpecification],
        strategy: BalancingStrategy,
        params: Optional[ModelParams],
        cv: int,
        optimization_strategy: OptimizationStrategy,
        task_type: TaskType,
        **kwargs,
    ) -> ModelSpecification:
        """Instantiate a model specification with shared initialization args.

        Args:
            model_cls: Concrete model specification class.
            strategy: Balancing strategy.
            params: Optional params instance to pass through.
            cv: Cross-validation fold count.
            optimization_strategy: Grid/random/manual strategy.
            task_type: Task type for the model.
            **kwargs: Additional constructor args.

        Returns:
            Instantiated model specification.
        """
        init_kwargs = {
            "balancing_strategy": strategy,
            "params": params,
            "cv": cv,
            "optimization_strategy": optimization_strategy,
            "task_type": task_type,
            **kwargs,
        }
        return model_cls(**init_kwargs)

    @classmethod
    def create_model_from_config(
        cls, config: ModelConfiguration
    ) -> Optional[ModelSpecification]:
        """Instantiate a model instance from a ModelConfiguration.

        Args:
            config: Configuration containing model type and parameters.

        Returns:
            Model instance or None if the type is not resolvable.
        """
        # Find the corresponding class from the model_type enum
        model_cls = config.model_type.model_class

        if model_cls is not None:
            # Instantiate using the extracted method
            return ModelSpecification.instantiate_model(
                model_cls=model_cls,
                strategy=config.balancing_strategy,
                params=config.model_params,  # Assuming this fits the ModelParams type
                cv=config.cv,
                optimization_strategy=config.optimization_strategy,
                task_type=config.task_type,
            )
        else:
            print(f"Unable to instantiate model class: {model_cls}")
            return None
