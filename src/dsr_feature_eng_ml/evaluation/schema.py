"""Evaluation schema models and feature metadata structures."""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, Optional, Sequence, Tuple, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dsr_utils.formatting import (
    BoolFormat,
    CurrencyFormat,
    DataFormat,
    DateTimeFormat,
    EnumFormat,
    FloatFormat,
    FormatConfig,
    IntegerFormat,
    PercentageFormat,
    StringFormat,
    ValueDescFormat,
    format_label_value_pairs,
)
from pandas.api.extensions import ExtensionDtype
from scipy.stats import kurtosis, skew
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelGeneralization,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.prefs_instance import prefs
from dsr_feature_eng_ml.utils.memory import validate_n_jobs

if TYPE_CHECKING:
    from dsr_feature_eng_ml.models.model_specification import (
        ModelParams,
        ModelSpecification,
    )

T_Params = TypeVar("T_Params", bound="ModelParams")


def _f_audit_gap(gap: Optional[float], status: ModelGeneralization) -> str:
    """Private domain-specific wrapper for the audit report."""
    audit_gap_format = ValueDescFormat(
        precision=prefs.score_format.precision, description=status.value
    )
    return audit_gap_format.format_value(gap)


class DatasetFormatters:
    """
    Container for dtype-specific formatting rules used in reports.

    Centralizes the formatter selection logic so tables and charts can render
    values consistently based on pandas dtypes.
    """

    def __init__(
        self,
        dtype_float: (
            CurrencyFormat
            | PercentageFormat
            | IntegerFormat
            | FloatFormat
            | ValueDescFormat
            | DateTimeFormat
            | DataFormat
        ) = FloatFormat(precision=2),
        dtype_object: EnumFormat | StringFormat = StringFormat(),
        dtype_int: IntegerFormat | ValueDescFormat = IntegerFormat(),
        dtype_bool: BoolFormat = BoolFormat(),
        dtype_datetime: DateTimeFormat = DateTimeFormat(
            date_format="%m-%d-%Y", time_format="%H:%M:%S"
        ),
        dtype_timedelta: DateTimeFormat = DateTimeFormat(use_duration_format=True),
        dtype_category: StringFormat = StringFormat(),
    ):
        """Initialize dataset formatter mappings for common dtypes."""
        self._dtype_float = dtype_float
        self._dtype_object = dtype_object
        self._dtype_int = dtype_int
        self._dtype_bool = dtype_bool
        self._dtype_datetime = dtype_datetime
        self._dtype_timedelta = dtype_timedelta
        self._dtype_category = dtype_category

    @property
    def dtype_float(
        self,
    ) -> (
        CurrencyFormat
        | PercentageFormat
        | IntegerFormat
        | FloatFormat
        | ValueDescFormat
        | DateTimeFormat
        | DataFormat
    ):
        """Formatter used for float-like numeric dtypes."""
        return self._dtype_float

    @dtype_float.setter
    def dtype_float(
        self,
        val: (
            CurrencyFormat
            | PercentageFormat
            | IntegerFormat
            | FloatFormat
            | ValueDescFormat
            | DateTimeFormat
            | DataFormat
        ),
    ) -> None:
        self._dtype_float = val

    @property
    def dtype_object(self) -> EnumFormat | StringFormat:
        """Formatter used for object and string dtypes."""
        return self._dtype_object

    @dtype_object.setter
    def dtype_object(self, val: EnumFormat | StringFormat) -> None:
        self._dtype_object = val

    @property
    def dtype_int(self) -> IntegerFormat | ValueDescFormat:
        """Formatter used for integer dtypes."""
        return self._dtype_int

    @dtype_int.setter
    def dtype_int(self, val: IntegerFormat | ValueDescFormat) -> None:
        self._dtype_int = val

    @property
    def dtype_bool(self) -> BoolFormat:
        """Formatter used for boolean dtypes."""
        return self._dtype_bool

    @dtype_bool.setter
    def dtype_bool(self, val: BoolFormat) -> None:
        self._dtype_bool = val

    @property
    def dtype_datetime(self) -> DateTimeFormat:
        """Formatter used for datetime dtypes."""
        return self._dtype_datetime

    @dtype_datetime.setter
    def dtype_datetime(self, val: DateTimeFormat) -> None:
        self._dtype_datetime = val

    @property
    def dtype_timedelta(self) -> DateTimeFormat:
        """Formatter used for timedelta/duration dtypes."""
        return self._dtype_timedelta

    @dtype_timedelta.setter
    def dtype_timedelta(self, val: DateTimeFormat) -> None:
        self._dtype_timedelta = val

    @property
    def dtype_category(self) -> StringFormat:
        """Formatter used for categorical dtypes."""
        return self._dtype_category

    @dtype_category.setter
    def dtype_category(self, val: StringFormat) -> None:
        self._dtype_category = val

    def fmt_for_dtype(
        self, input_dtype: np.dtype[Any] | ExtensionDtype
    ) -> FormatConfig:
        """
        Return the best-matching formatter for a pandas dtype.

        Args:
            input_dtype: Pandas or NumPy dtype to classify.

        Returns:
            Formatter instance appropriate for the given dtype.
        """
        # Handle ExtensionDtype first (CategoricalDtype, etc.)
        if isinstance(input_dtype, pd.CategoricalDtype):
            return self.dtype_category

        # Check for datetime variants using pandas API
        if pd.api.types.is_datetime64_any_dtype(input_dtype):
            return self.dtype_datetime

        if pd.api.types.is_timedelta64_dtype(input_dtype):
            return self.dtype_timedelta

        # Handle standard NumPy dtypes
        if isinstance(input_dtype, np.dtype):
            if np.isdtype(input_dtype, "integral"):
                return self.dtype_int
            if np.isdtype(input_dtype, "real floating"):
                return self.dtype_float
            if np.isdtype(input_dtype, "bool"):
                return self.dtype_bool

        return self.dtype_object


class FeatureMetadata:
    """
    Metadata describing an input feature and its reporting configuration.

    Captures display and formatting context while tracking inclusion in model
    fitting and parent/child feature relationships for reporting.
    """

    def __init__(
        self,
        name: str,
        id: str,
        position: int,
        short_name: str | None = None,
        formatter: FormatConfig = StringFormat(),
        description: str = "",
        is_used_in_fit: bool = True,
        parent_name: str | None = None,
    ):
        """Initialize metadata for a dataset feature."""
        self._name = name
        self._id = id
        self._position = position
        self._short_name = short_name if short_name is not None else name
        self._formatter = formatter
        self._description = description
        self._is_used_in_fit = is_used_in_fit
        self._parent_name = parent_name

    @property
    def name(self) -> str:
        """Raw feature/column name."""
        return self._name

    @property
    def id(self) -> str:
        """Stable feature identifier used in reports (e.g., 'F01')."""
        return self._id

    @property
    def position(self) -> int:
        """Zero-based index of the feature in the source DataFrame."""
        return self._position

    @property
    def short_name(self) -> str:
        """Display-friendly short name for charts and tables."""
        return self._short_name

    @short_name.setter
    def short_name(self, val: str) -> None:
        self._short_name = val

    @property
    def formatter(self) -> FormatConfig:
        """Formatter used when rendering this feature in reports."""
        return self._formatter

    @formatter.setter
    def formatter(self, val: FormatConfig) -> None:
        self._formatter = val

    @property
    def description(self) -> str:
        """Optional free-form description for reporting context."""
        return self._description

    @description.setter
    def description(self, val: str) -> None:
        self._description = val

    @property
    def is_used_in_fit(self) -> bool:
        """True if the feature should be included in model training."""
        return self._is_used_in_fit

    @is_used_in_fit.setter
    def is_used_in_fit(self, val: bool) -> None:
        self._is_used_in_fit = val

    @property
    def parent_name(self) -> str | None:
        """Name of the parent feature used for consolidated reporting."""
        return self._parent_name

    @parent_name.setter
    def parent_name(self, val: str | None) -> None:
        self._parent_name = val

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature metadata to a dictionary for JSON exports."""
        return {
            "name": self.name,
            "id": self.id,
            "position": self.position,
            "short_name": self.short_name,
            "description": self.description,
            "formatter": self.formatter.to_dict(),
            "is_used_in_fit": self.is_used_in_fit,
            "parent_name": self.parent_name,
        }

    @classmethod
    def dict_to_set(
        cls, feature_dict: dict[str, FeatureMetadata], target_column: str
    ) -> set[FeatureMetadata]:
        """Convert a feature metadata mapping to a filtered set for fitting."""
        return cls.list_to_set(list(feature_dict.values()), target_column)

    @classmethod
    def list_to_set(
        cls, feature_list: list[FeatureMetadata], target_column: str
    ) -> set[FeatureMetadata]:
        """Filter feature metadata list to eligible training features."""
        return {f for f in feature_list if f.is_used_in_fit and f.name != target_column}

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        formatters: DatasetFormatters = DatasetFormatters(),
        format_exceptions: dict[str, FormatConfig] | None = None,
        feature_parent: dict[str, str] | None = None,
        exclude_from_fit: set[str] | None = None,
        short_names: dict[str, str] | None = None,
    ) -> dict[str, FeatureMetadata]:
        """
        Build feature metadata mapping for every column in a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The source data.
        formatters : DatasetFormatters
            Registry for dtype-based formatting.
        format_exceptions : dict[str, FormatConfig], optional
            Per-column formatter overrides.
        feature_parent : dict[str, str], optional
            Mapping of column name to parent feature name.
        exclude_from_fit : set[str], optional
            Columns to exclude from training.
        short_names : dict[str, str], optional
            Mapping of column name to display-friendly label.
        """
        # Initialize defaults for optional dicts/sets
        exceptions = format_exceptions or {}
        parents = feature_parent or {}
        exclude = exclude_from_fit or set()
        shorts = short_names or {}

        fm_dict: dict[str, FeatureMetadata] = {}
        all_cols = set(df.columns)
        padding = max(len(str(len(df.columns))), 2)
        id_fmt = IntegerFormat(width=padding, pad_value="0")

        for i, col in enumerate(df.columns):
            # 1. Determine Formatter
            fmt = exceptions.get(col) or formatters.fmt_for_dtype(df[col].dtype)

            # 2. Assign Stable ID
            feature_id = f"F{id_fmt.format_value(i+1)}"

            # 3. Validate Parent Relationship
            parent = parents.get(col)
            if parent and parent not in all_cols:
                # Use a logging-style print or actual logger here if available
                print(f"WARNING: Feature '{col}' has invalid parent '{parent}'.")
                parent = None

            # 4. Create Metadata Instance
            fm_dict[col] = FeatureMetadata(
                name=col,
                id=feature_id,
                position=i,
                formatter=fmt,
                short_name=shorts.get(col, col),
                is_used_in_fit=(col not in exclude),
                parent_name=parent,
            )

        return fm_dict


class ModelFeatureImportance:
    """
    Analyzes and manages feature importance from fitted models.

    Calculates feature importance scores, cumulative importance, and identifies
    optimal feature subsets based on importance thresholds.

    Attributes
    ----------
    features : list[str]
        Feature names sorted by importance (descending).
    feature_importances : pd.DataFrame
        DataFrame with columns 'feature', 'importance', 'id', and
        'cumulative_importance'.
    threshold_80_idx : int
        Index of the first feature reaching 80% cumulative importance.
    threshold_95_idx : int
        Index of the first feature reaching 95% cumulative importance.
    """

    def __init__(
        self,
        feature_set: set[FeatureMetadata],
        importances: np.ndarray,
    ):
        """Initialize feature importance data and calculate cumulative values."""
        # Create the dataframe using the array and metadata
        self.feature_importances = pd.DataFrame(
            {
                "feature": [f.name for f in feature_set],
                "importance": importances,
                "id": [f.id for f in feature_set],
            }
        ).sort_values("importance", ascending=False)

        # Calculate cumulative values
        self.feature_importances["cumulative_importance"] = self.feature_importances[
            "importance"
        ].cumsum()

        # Ensure consistent floating point types for serialization
        self.feature_importances = self.feature_importances.astype(
            {
                "importance": "float32",
                "cumulative_importance": "float32",
            }
        )

        self.features = self.feature_importances["feature"].to_list()
        self.threshold_80_idx = 0
        self.threshold_95_idx = 0

    @property
    def get_feature_column_index(self) -> int:
        """Return the integer index of the 'feature' column."""
        idx = self.feature_importances.columns.get_indexer_for(pd.Index(["feature"]))
        if idx.size != 1 or idx[0] == -1:
            raise ValueError("Expected exactly one 'feature' column.")
        return int(idx[0])

    @property
    def get_importance_column_index(self) -> int:
        """Return the integer index of the 'importance' column."""
        idx = self.feature_importances.columns.get_indexer_for(pd.Index(["importance"]))
        if idx.size != 1 or idx[0] == -1:
            raise ValueError("Expected exactly one 'importance' column.")
        return int(idx[0])

    @property
    def get_cumulative_importance_column_index(self) -> int:
        """Return the integer index of the 'cumulative_importance' column."""
        idx = self.feature_importances.columns.get_indexer_for(
            pd.Index(["cumulative_importance"])
        )
        if idx.size != 1 or idx[0] == -1:
            raise ValueError("Expected exactly one 'cumulative_importance' column.")
        return int(idx[0])

    @classmethod
    def empty(cls) -> ModelFeatureImportance:
        """Create an empty feature importance instance."""
        return cls(feature_set=set(), importances=np.array([]))

    def to_dict(self, include_full_df: bool = True) -> dict[str, Any]:
        """Convert importance data to a dictionary for serialization."""
        data = {
            "features": self.features,
            "threshold_80_idx": self.threshold_80_idx,
            "threshold_95_idx": self.threshold_95_idx,
        }

        if include_full_df:
            # records orientation is best for JSON row-based exports
            data["feature_importances"] = self.feature_importances.to_dict(
                orient="records"
            )

        return data

    def info(self) -> str:
        """Return a formatted summary of feature importances."""
        retval: str = ""
        for i, row in enumerate(self.feature_importances.itertuples()):
            retval += "{:<3} {:<20} Importance: {:.4f}   {:>8.2%}\n".format(
                i + 1, row.feature, row.importance, row.cumulative_importance
            )
        return retval

    def calc_threshold_indices(self) -> None:
        """
        Calculate index positions for 80% and 95% cumulative thresholds.

        These indices define the minimum set of features required to explain
        the majority of the model's predictive power.
        """
        self.threshold_80_idx = 0
        self.threshold_95_idx = 0
        feature_count = len(self.feature_importances)

        # Scan for threshold cross-over points
        for n in range(1, feature_count + 1):
            # Using iat for scalar performance during the loop
            cumulative = self.feature_importances.iloc[n - 1]["cumulative_importance"]

            if self.threshold_80_idx == 0 and cumulative >= 0.8:
                self.threshold_80_idx = n

            if self.threshold_95_idx == 0 and cumulative >= 0.95:
                # Set +1 to include the feature that pushed it over the limit
                self.threshold_95_idx = n + 1
                break

        # Safety clamp to ensure indices do not exceed the feature count
        if self.threshold_95_idx > feature_count or self.threshold_95_idx == 0:
            self.threshold_95_idx = feature_count
        if self.threshold_80_idx == 0:
            self.threshold_80_idx = feature_count


@dataclass(frozen=True)
class DataSplits:
    """
    Immutable container for train/validation/test data splits in workflows.

    This dataclass encapsulates dataset splits and provides factory methods for
    creating balanced training data. All instances are immutable.

    Attributes
    ----------
    features_to_include : list[str]
        Column names of features to include in the dataset.
    target_column : str
        Name of the target variable column.
    test_features : pd.DataFrame
        Test set features for final model evaluation.
    test_target : pd.Series
        Test set target values for final model evaluation.
    train_features : pd.DataFrame
        Training set features for model fitting.
    train_target : pd.Series
        Training set target values for model fitting.
    val_features : pd.DataFrame
        Validation set features for hyperparameter tuning.
    val_target : pd.Series
        Validation set target values for hyperparameter tuning.
    original_row_count : int
        Row count of the source dataset before any splitting.
    random_state : Optional[int]
        Random seed for reproducible operations.
    scaler : StandardScaler, optional
        The fitted scaler used for numerical features.
    """

    features_to_include: list[str]
    target_column: str
    test_features: pd.DataFrame
    test_target: pd.Series
    train_features: pd.DataFrame
    train_target: pd.Series
    val_features: pd.DataFrame
    val_target: pd.Series
    original_row_count: int
    random_state: Optional[int]
    scaler: StandardScaler | None = None

    @property
    def evaluation_features(self) -> list[str]:
        """Return the list of columns available for evaluation."""
        return self.val_features.columns.tolist()

    @classmethod
    def from_data_source(
        cls,
        src: pd.DataFrame,
        features_to_include: list[str],
        target_column: str,
        test_size: float,
        valid_size: float,
        original_row_count: int,
        random_state: Optional[int],
        scale_features: bool = True,
        shuffle: bool = True,
        stratify: bool = False,
    ) -> DataSplits:
        """
        Create DataSplits from a source DataFrame with automatic splitting.

        Parameters
        ----------
        src : pd.DataFrame
            Source DataFrame containing features and target.
        features_to_include : list[str]
            Column names to use as features.
        target_column : str
            Name of the target variable column.
        test_size : float
            Proportion of data for test set (0.0 to 1.0).
        valid_size : float
            Proportion of main data for validation (0.0 to 1.0).
        original_row_count : int
            Baseline row count for metadata tracking.
        random_state : Optional[int]
            Random seed for reproducibility.
        scale_features : bool, default True
            Whether to apply StandardScaler to numeric features.
        shuffle : bool, default True
            Whether to shuffle data before splitting.
        stratify : bool, default False
            Whether to use stratified splitting based on target.
        """
        target = src[target_column]
        features = src[features_to_include]

        # 1. Split Main (Train+Val) and Test
        strat_p = target if stratify else None
        main_feat, test_feat, main_targ, test_targ = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=strat_p,
        )

        # 2. Split Main into Train and Validation
        strat_v = main_targ if stratify else None
        train_feat, val_feat, train_targ, val_targ = train_test_split(
            main_feat,
            main_targ,
            test_size=valid_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=strat_v,
        )

        scaler_to_store: StandardScaler | None = None

        # 3. Handle Scaling (Numeric Only)
        if scale_features:
            numeric_cols = train_feat.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            categorical_cols = train_feat.select_dtypes(
                exclude=[np.number]
            ).columns.tolist()

            scaler = StandardScaler()

            # Fit only on training data to prevent leakage
            train_scaled = scaler.fit_transform(train_feat[numeric_cols])
            val_scaled = scaler.transform(val_feat[numeric_cols])
            test_scaled = scaler.transform(test_feat[numeric_cols])

            # Reconstruction while preserving index
            train_feat_scaled = pd.DataFrame(
                train_scaled, columns=numeric_cols, index=train_feat.index
            )
            val_feat_scaled = pd.DataFrame(
                val_scaled, columns=numeric_cols, index=val_feat.index
            )
            test_feat_scaled = pd.DataFrame(
                test_scaled, columns=numeric_cols, index=test_feat.index
            )

            if categorical_cols:
                train_feat = pd.concat(
                    [train_feat_scaled, train_feat[categorical_cols]], axis=1
                )
                val_feat = pd.concat(
                    [val_feat_scaled, val_feat[categorical_cols]], axis=1
                )
                test_feat = pd.concat(
                    [test_feat_scaled, test_feat[categorical_cols]], axis=1
                )
            else:
                train_feat, val_feat, test_feat = (
                    train_feat_scaled,
                    val_feat_scaled,
                    test_feat_scaled,
                )

            # Maintain original column order
            train_feat = train_feat[features_to_include]
            val_feat = val_feat[features_to_include]
            test_feat = test_feat[features_to_include]
            scaler_to_store = scaler

        return cls(
            features_to_include=features_to_include,
            target_column=target_column,
            test_features=test_feat,
            test_target=test_targ,
            train_features=train_feat,
            train_target=train_targ,
            val_features=val_feat,
            val_target=val_targ,
            original_row_count=original_row_count,
            random_state=random_state,
            scaler=scaler_to_store,
        )

    @classmethod
    def from_data_splits(
        cls,
        src: DataSplits,
        features_to_include: list[str],
    ) -> DataSplits:
        """
        Create a new DataSplits instance with a subset of features.

        This method is particularly useful during iterative feature selection,
        allowing you to prune the dataset while maintaining the original
        splitting boundaries and target values.

        Parameters
        ----------
        src : DataSplits
            The source instance to derive the subset from.
        features_to_include : list[str]
            The specific feature columns to retain in the new instance.

        Returns
        -------
        DataSplits
            A new instance containing only the requested features.
        """
        # We explicitly use pd.DataFrame() wrapper around the slice to ensure
        # Pylance recognizes the return type and avoids SettingWithCopy warnings.
        return cls(
            features_to_include=features_to_include,
            target_column=src.target_column,
            test_features=pd.DataFrame(src.test_features[features_to_include]),
            test_target=src.test_target.copy(),
            train_features=pd.DataFrame(src.train_features[features_to_include]),
            train_target=src.train_target.copy(),
            val_features=pd.DataFrame(src.val_features[features_to_include]),
            val_target=src.val_target.copy(),
            original_row_count=src.original_row_count,
            random_state=src.random_state,
            scaler=src.scaler,
        )

    @classmethod
    def empty(cls) -> DataSplits:
        """
        Create an empty DataSplits instance for placeholder initialization.

        Returns
        -------
        DataSplits
            An instance with empty DataFrames and Series.
        """
        empty_df = pd.DataFrame()
        # Explicit dtype on Series prevents future pandas deprecation warnings
        # regarding object-dtype inference.
        empty_series = pd.Series(dtype="float64")

        return cls(
            features_to_include=[],
            target_column="",
            test_features=empty_df,
            test_target=empty_series,
            train_features=empty_df,
            train_target=empty_series,
            val_features=empty_df,
            val_target=empty_series,
            original_row_count=0,
            random_state=0,
            scaler=None,
        )

    def info(self) -> None:
        """
        Display technical summaries of all data splits.

        Prints structural information (dtypes, non-null counts, memory usage)
        for the train, validation, and test feature sets.
        """
        # We group these to provide a clear sequence of the split lifecycle
        print("--- Training Set ---")
        self.train_features.info()
        print("\n--- Validation Set ---")
        self.val_features.info()
        print("\n--- Test Set ---")
        self.test_features.info()

    def inverse_transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Revert scaled numerical features to their original units.

        Uses the internal StandardScaler to transform numerical columns back
        to their baseline distribution for reporting and display.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing scaled features.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with identified numeric columns
            reverted to original units.
        """
        if self.scaler is None:
            return df.copy()

        # Create a deep copy to ensure immutability of the source split
        df_inv = df.copy()

        # Identify numeric candidates for inversion
        numeric_cols = df_inv.select_dtypes(include=[np.number]).columns.tolist()

        # Cross-reference with columns the scaler was actually fitted on
        # This prevents errors if the DataFrame contains new or unscaled numeric columns.
        scaled_feature_names = list(self.scaler.feature_names_in_)
        cols_to_revert = [c for c in numeric_cols if c in scaled_feature_names]

        if cols_to_revert:
            # Revert the values in place on the copied DataFrame
            df_inv[cols_to_revert] = self.scaler.inverse_transform(
                df_inv[cols_to_revert]
            )

        return df_inv

    def with_upsampled_training(
        self, X: pd.DataFrame, y: pd.Series[Any], use_combined_data: bool = False
    ) -> DataSplits:
        """
        Return a new DataSplits instance with perfectly balanced upsampled training data.
        """
        # 1. Segment dataset by binary class
        feat_zeros = X[y == 0]
        feat_ones = X[y == 1]
        targ_zeros = y[y == 0]
        targ_ones = y[y == 1]

        n0, n1 = len(targ_zeros), len(targ_ones)

        # 2. Identify minority vs majority
        if n0 > n1:
            feat_min, feat_maj = feat_ones, feat_zeros
            targ_min, targ_maj = targ_ones, targ_zeros
        else:
            feat_min, feat_maj = feat_zeros, feat_ones
            targ_min, targ_maj = targ_zeros, targ_ones

        # 3. Exact Balancing Logic: Sample minority class to match majority count
        # This replaces the 'factor' logic to ensure counts[0] == counts[1]
        feat_min_up = feat_min.sample(
            len(feat_maj), replace=True, random_state=self.random_state
        )
        targ_min_up = targ_min.sample(
            len(targ_maj), replace=True, random_state=self.random_state
        )

        feat_upsampled = pd.concat([feat_maj, feat_min_up])
        targ_upsampled = pd.concat([targ_maj, targ_min_up])

        # 4. Shuffle to prevent class grouping
        shuffled_f, shuffled_t = cast(
            Tuple[pd.DataFrame, pd.Series],
            shuffle(feat_upsampled, targ_upsampled, random_state=self.random_state),
        )

        # 5. Reconstruct objects
        train_feat_up = pd.DataFrame(shuffled_f, columns=feat_upsampled.columns)
        train_targ_up = pd.Series(
            shuffled_t, name=targ_upsampled.name, index=train_feat_up.index
        )

        return DataSplits(
            features_to_include=self.features_to_include,
            target_column=self.target_column,
            test_features=self.test_features,
            test_target=self.test_target,
            train_features=train_feat_up,
            train_target=train_targ_up,
            val_features=pd.DataFrame() if use_combined_data else self.val_features,
            val_target=(
                pd.Series(dtype="float64") if use_combined_data else self.val_target
            ),
            original_row_count=self.original_row_count,
            random_state=self.random_state,
            scaler=self.scaler,
        )

    def with_downsampled_training(
        self, X: pd.DataFrame, y: pd.Series[Any], use_combined_data: bool = False
    ) -> DataSplits:
        """
        Return a new DataSplits instance with perfectly balanced downsampled training data.
        """
        # 1. Segment dataset by binary class
        feat_zeros = X[y == 0]
        feat_ones = X[y == 1]
        targ_zeros = y[y == 0]
        targ_ones = y[y == 1]
        n0, n1 = len(targ_zeros), len(targ_ones)

        # 2. Identify minority vs majority
        if n0 > n1:
            feat_min, feat_maj = feat_ones, feat_zeros
            targ_min, targ_maj = targ_ones, targ_zeros
        else:
            feat_min, feat_maj = feat_zeros, feat_ones
            targ_min, targ_maj = targ_zeros, targ_ones

        # 3. Exact Balancing Logic: Sample majority to match minority count
        # Using n=len(feat_min) prevents fractional rounding discrepancies.
        feat_downsampled = pd.concat(
            [feat_min, feat_maj.sample(n=len(feat_min), random_state=self.random_state)]
        )
        targ_downsampled = pd.concat(
            [targ_min, targ_maj.sample(n=len(targ_min), random_state=self.random_state)]
        )

        # 4. Shuffle and cast
        from typing import Tuple

        shuffled_f, shuffled_t = cast(
            Tuple[pd.DataFrame, pd.Series],
            shuffle(feat_downsampled, targ_downsampled, random_state=self.random_state),
        )

        # 5. Reconstruct pandas objects
        train_feat_down = pd.DataFrame(shuffled_f, columns=feat_downsampled.columns)
        train_targ_down = pd.Series(
            shuffled_t, name=targ_downsampled.name, index=train_feat_down.index
        )

        return DataSplits(
            features_to_include=self.features_to_include,
            target_column=self.target_column,
            test_features=self.test_features,
            test_target=self.test_target,
            train_features=train_feat_down,
            train_target=train_targ_down,
            val_features=pd.DataFrame() if use_combined_data else self.val_features,
            val_target=(
                pd.Series(dtype="float64") if use_combined_data else self.val_target
            ),
            original_row_count=self.original_row_count,
            random_state=self.random_state,
            scaler=self.scaler,
        )

    def get_balanced_train_data(
        self,
        strategy: BalancingStrategy,
        feature_set: set[FeatureMetadata],
        use_combined_data: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series[Any]]:
        """
        Retrieve training features and targets aligned with the requested balancing strategy.

        This method handles the selection of data (Training vs. Combined) and
        applies the appropriate resampling technique (Oversampling vs. Undersampling)
        to ensure the model fits on the intended class distribution.

        Parameters
        ----------
        strategy : BalancingStrategy
            The resampling strategy to apply (NONE, OVERSAMPLED, or UNDERSAMPLED).
        feature_set : set[FeatureMetadata]
            The specific features to include in the returned DataFrames.
        use_combined_data : bool, default False
            If True, merges training and validation sets before balancing.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series[Any]]
            A tuple of (features, targets) ready for model training.
        """
        # 1. Define the base feature list from metadata
        feature_list = [f.name for f in feature_set]

        # 2. Source Data Selection
        if use_combined_data:
            # Merge Train and Val splits for final fitting
            X = pd.concat(
                [self.train_features[feature_list], self.val_features[feature_list]]
            )
            y = pd.concat([self.train_target, self.val_target])
        else:
            # Use only the training split
            X = self.train_features[feature_list]
            y = self.train_target

        # 3. Apply Balancing Strategy
        if strategy == BalancingStrategy.OVERSAMPLED:
            # Returns a new DataSplits instance with duplicated minority samples
            balanced_splits = self.with_upsampled_training(X, y, use_combined_data)
            return balanced_splits.train_features, balanced_splits.train_target

        if strategy == BalancingStrategy.UNDERSAMPLED:
            # Returns a new DataSplits instance with reduced majority samples
            balanced_splits = self.with_downsampled_training(X, y, use_combined_data)
            return balanced_splits.train_features, balanced_splits.train_target

        # Default: No balancing (BalancingStrategy.NONE)
        return X, y

    def get_train_weights(
        self, strategy: BalancingStrategy, is_regression: bool = True
    ) -> np.ndarray | None:
        """
        Calculate sample weights for training data to address class or value imbalance.

        For classification, weights are based on inverse class frequency. For regression,
        weights are derived from inverse bin frequency using histogram binning with
        smoothing to handle rare continuous values.

        Parameters
        ----------
        strategy : BalancingStrategy
            The balancing strategy; weights are only returned if set to WEIGHTED.
        is_regression : bool, default True
            Whether the current task is regression.

        Returns
        -------
        np.ndarray | None
            An array of normalized sample weights, or None if the strategy is not WEIGHTED.
        """
        if strategy != BalancingStrategy.WEIGHTED:
            return None

        y = self.train_target

        if is_regression:
            # 1. Use histogram binning for continuous target weighting
            counts, bin_edges = np.histogram(y, bins=20)

            # 2. Smoothing: Add a constant (k) to prevent extreme weights for outlier bins.
            # We use the mean count per bin as the smoothing constant.
            k = len(y) / 20
            smoothed_counts = counts + k

            # 3. Map samples to smoothed inverse frequencies
            bin_indices = np.digitize(y, bin_edges[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
            weights = 1.0 / smoothed_counts[bin_indices]
        else:
            # 4. Classification: Inverse class frequency mapping
            class_counts = y.value_counts()
            weights = y.map(1.0 / class_counts)

        # 5. Normalize weights (mean=1.0) to improve optimizer stability
        normalized_weights = weights / weights.mean()

        # 6. Ensure return type consistency for Scikit-Learn
        if isinstance(normalized_weights, pd.Series):
            return cast(np.ndarray, normalized_weights.to_numpy())

        return cast(np.ndarray, normalized_weights)

    def with_feature_subset(self, feature_subset: list[str]) -> DataSplits:
        """
        Create a new DataSplits instance with a subset of features.

        This memory-efficient factory method creates new DataFrames only for
        the selected features while reusing existing target Series objects.

        Parameters
        ----------
        feature_subset : list[str]
            List of feature column names to include in the new split.

        Returns
        -------
        DataSplits
            A new instance with filtered features and shared target data.
        """
        # Wrapping the slice in pd.DataFrame() ensures a fresh object for features,
        # preventing SettingWithCopy warnings in downstream engineering steps.
        return DataSplits(
            features_to_include=feature_subset,
            target_column=self.target_column,
            test_features=pd.DataFrame(self.test_features[feature_subset]),
            test_target=self.test_target,  # Shared reference for memory efficiency
            train_features=pd.DataFrame(self.train_features[feature_subset]),
            train_target=self.train_target,  # Shared reference
            val_features=pd.DataFrame(self.val_features[feature_subset]),
            val_target=self.val_target,  # Shared reference
            original_row_count=self.original_row_count,
            random_state=self.random_state,
            scaler=self.scaler,
        )

    def auc_roc_curve(
        self, test_proba: np.ndarray, plot_title: str = "ROC Curve"
    ) -> float:
        """
        Plot ROC curve and calculate AUC score for binary classification.

        Generates a Receiver Operating Characteristic (ROC) curve and calculates
        the Area Under the Curve (AUC) to assess classification performance.

        Parameters
        ----------
        test_proba : np.ndarray
            Predicted probabilities for the positive class on the test set.
        plot_title : str, default "ROC Curve"
            The title displayed on the generated chart.

        Returns
        -------
        float
            The calculated AUC score (ranging from 0.0 to 1.0).
        """
        # Calculate FPR and TPR for various thresholds
        fpr, tpr, thresholds = roc_curve(self.test_target, test_proba)
        auc_score = float(auc(fpr, tpr))
        auc_score_format = FloatFormat(precision=4)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            label=f"Model (AUC = {auc_score_format.format_value(auc_score)})",
            linewidth=2,
        )

        # Reference line for a random (no-skill) classifier
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier"
        )

        # Threshold Annotation Logic
        threshold_format = FloatFormat(precision=2)
        # Select key indices to prevent over-cluttering the plot
        indices = [0, len(thresholds) // 4, len(thresholds) // 2, len(thresholds) - 1]

        for idx, i in enumerate(indices):
            plt.scatter(fpr[i], tpr[i], color="red", s=50, zorder=5)
            plt.annotate(
                f"{threshold_format.format_value(thresholds[i])}",
                xy=(fpr[i], tpr[i]),
                xytext=(10, 10 + idx * 15),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

        # Plot Aesthetics
        plt.xlim([0.0, 1.0])
        plt.ylim(
            [0.0, 1.05]
        )  # Slightly higher than 1.0 to ensure the top line isn't cut off
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title(plot_title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

        return auc_score


@dataclass
class ModelConfigurationStats:
    """
    Statistical summaries for model train/val/test target distributions.

    This class captures descriptive statistics for each split and calculates
    drift metrics to assess data consistency.
    """

    @dataclass
    class ModelSplitStats:
        """Descriptive stats for a single data split."""

        class SplitType(Enum):
            TRAIN_VAL = auto()
            TEST = auto()

        mean: float
        std: float
        median: float
        skew: float
        kurtosis: float

    model_split_stats: dict[str, ModelConfigurationStats.ModelSplitStats]

    # Metrics valid primarily for SplitType.TRAIN_VAL
    quality_score: float
    drift_index: float
    mean_delta: float
    std_delta: float

    @classmethod
    def from_config(
        cls,
        data_splits: DataSplits,
        config: ModelConfiguration,
        split_type: ModelSplitStats.SplitType = ModelSplitStats.SplitType.TRAIN_VAL,
    ) -> ModelConfigurationStats:
        """
        Build split statistics and drift metrics from a model configuration.

        Parameters
        ----------
        data_splits : DataSplits
            Dataset splits containing target vectors.
        config : ModelConfiguration
            Configuration providing performance scores for delta logic.
        split_type : SplitType
            Determines whether to process TRAIN_VAL or TEST splits.
        """

        def get_stats(
            target_series: pd.Series,
        ) -> ModelConfigurationStats.ModelSplitStats:
            """Helper to calculate consistent stats for any series."""
            arr = target_series.to_numpy()
            return ModelConfigurationStats.ModelSplitStats(
                mean=float(target_series.mean()),
                std=float(target_series.std()),
                median=float(target_series.median()),
                skew=float(skew(arr)),
                kurtosis=float(kurtosis(arr)),
            )

        match split_type:
            case ModelConfigurationStats.ModelSplitStats.SplitType.TRAIN_VAL:
                train_stats = get_stats(data_splits.train_target)
                val_stats = get_stats(data_splits.val_target)

                # Quality Score Logic
                # penalizes models if the 'cleaned' score exceeds the raw score
                c_score = config.score_val_cleaned or 0.0
                r_score = config.score_val or 0.0

                if c_score <= r_score:
                    quality_score = 100.0
                else:
                    gap = c_score - r_score
                    penalty = prefs.get_penalty_multiplier_for_task_type(
                        config.task_type
                    )
                    quality_score = 100.0 - (gap * penalty)

                # Drift and Delta Logic
                drift_index = (
                    abs(train_stats.mean - val_stats.mean) / train_stats.mean
                    if train_stats.mean != 0
                    else 0.0
                )

                score_cv = config.score_cv or 0.0
                mean_delta = r_score - score_cv

                std_delta = (
                    (abs(train_stats.std - val_stats.std) / train_stats.std * 100)
                    if train_stats.std != 0
                    else 0.0
                )

                return cls(
                    model_split_stats={"train": train_stats, "val": val_stats},
                    quality_score=quality_score,
                    drift_index=drift_index,
                    mean_delta=mean_delta,
                    std_delta=std_delta,
                )

            case ModelConfigurationStats.ModelSplitStats.SplitType.TEST:
                test_stats = get_stats(data_splits.test_target)
                return cls(
                    model_split_stats={"test": test_stats},
                    quality_score=0.0,
                    drift_index=0.0,
                    mean_delta=0.0,
                    std_delta=0.0,
                )


@dataclass(frozen=True)
@functools.total_ordering
class ModelConfiguration(Generic[T_Params]):
    """
    Frozen snapshot of a model run and its exhaustive evaluation metrics.

    This class serves as the immutable record for a single training and evaluation
    cycle, facilitating comparison in leaderboards and audit reports.
    """

    id: str
    model_type: ModelType
    task_type: TaskType
    balancing_strategy: BalancingStrategy
    optimization_strategy: OptimizationStrategy
    model_params: T_Params
    cv: int
    scoring: ScoringMetric
    n_jobs: int
    n_iter: int
    max_iter: int = 300
    has_val_set_evaluation_scores: bool = False
    has_test_set_evaluation_scores: bool = False
    use_combined_data: bool = False

    # Core performance metrics
    score_cv: float | None = None
    score_train: float | None = None
    score_val: float | None = None
    score_val_cleaned: float | None = None
    score_test: float | None = None

    # Regression-specific metrics
    mae_train: float | None = None
    mae_val: float | None = None
    mae_test: float | None = None
    mse_train: float | None = None
    mse_val: float | None = None
    mse_test: float | None = None
    r2_train: float | None = None
    r2_val: float | None = None
    r2_val_cleaned: float | None = None
    r2_test: float | None = None

    # Classification-specific metrics
    accuracy_train: float | None = None
    accuracy_val: float | None = None
    accuracy_val_cleaned: float | None = None
    accuracy_test: float | None = None

    # Prediction storage
    preds_val: pd.Series | None = None
    probs_val: pd.DataFrame | None = None
    preds_test: pd.Series | None = None
    probs_test: pd.DataFrame | None = None

    # Audit thresholds
    acceptable_gap: float = prefs.acceptable_gap
    large_gap: float = prefs.large_gap
    feature_analysis: ModelFeatureImportance = field(
        default_factory=ModelFeatureImportance.empty
    )

    # Telemetry and Resource tracking
    tuning_duration: float = 0.0
    fit_duration: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    estimated_peak_gb: float = 0.0
    actual_peak_gb: float = 0.0
    memory_risk_triggered: bool = False
    sampling_factor: float = 0.0
    concurrent_workers: int = 0
    model_multiplier: float = 1.0
    num_candidates: int = 1

    # Outlier handling
    filter_outliers: bool = False
    outlier_count: int = prefs.default_worst_errors_n
    efficiency_threshold: int = 0

    # Distribution Statistics
    train_mean: float = 0.0
    train_std: float = 0.0
    train_median: float = 0.0
    train_skew: float = 0.0
    train_kurtosis: float = 0.0
    val_mean: float = 0.0
    val_std: float = 0.0
    val_median: float = 0.0
    val_skew: float = 0.0
    val_kurtosis: float = 0.0
    test_mean: float = 0.0
    test_std: float = 0.0
    test_median: float = 0.0
    test_skew: float = 0.0
    test_kurtosis: float = 0.0

    # Audit Results
    mean_delta: float = 0.0
    std_delta: float = 0.0
    quality_score: float = 0.0
    drift_index: float = 0.0

    @property
    def r2_gap(self) -> float:
        """Absolute gap between train and validation R²."""
        if self.r2_train is None or self.r2_val is None:
            return 0.0
        return abs(self.r2_train - self.r2_val)

    @property
    def mae_gap(self) -> float:
        """Validation minus train MAE (positive indicates worse validation performance)."""
        if self.mae_train is None or self.mae_val is None:
            return 0.0
        return self.mae_val - self.mae_train

    @property
    def accuracy_gap(self) -> float:
        """Absolute difference between train and validation accuracy."""
        if self.accuracy_train is None or self.accuracy_val is None:
            return 0.0
        return abs(self.accuracy_train - self.accuracy_val)

    @property
    def gap(self) -> float:
        """The primary performance gap based on the specific task type."""
        if self.task_type == TaskType.REGRESSION:
            return self.r2_gap
        if self.task_type == TaskType.CLASSIFICATION:
            return self.accuracy_gap
        return 0.0

    @property
    def model_generalization(self) -> ModelGeneralization:
        """Classify the generalization status using primary gap metrics."""
        if self.score_train is None or self.score_val is None:
            return ModelGeneralization.PENDING

        current_gap = self.gap
        if current_gap > self.large_gap:
            return ModelGeneralization.OVERFIT
        if current_gap > self.acceptable_gap:
            return ModelGeneralization.MARGINAL
        return ModelGeneralization.WELL_FIT

    @property
    def params_dict(self) -> dict[str, Any]:
        """Dictionary representation of parameters for Scikit-Learn compatibility."""
        return self.model_params.to_dict()

    @property
    def total_duration(self) -> float:
        """Total runtime in seconds (tuning + fit)."""
        return self.tuning_duration + self.fit_duration

    @property
    def train_score(self) -> float:
        """Training score or 0.0 if unavailable."""
        return self.score_train or 0.0

    @property
    def val_score(self) -> float:
        """Validation score or 0.0 if unavailable."""
        return self.score_val or 0.0

    @property
    def test_score(self) -> float:
        """Test score or 0.0 if unavailable."""
        return self.score_test or 0.0

    def efficiency(self, data_splits: DataSplits) -> float:
        """
        Compute rows-per-second throughput for training and validation data.

        Parameters
        ----------
        data_splits : DataSplits
            The dataset splits used to determine row counts.

        Returns
        -------
        float
            Throughput in rows per second. Returns 0.0 if duration is 0.
        """
        if self.total_duration > 0.0:
            # Combine training and validation counts for a full lifecycle view
            total_rows = len(data_splits.train_features) + len(data_splits.val_features)
            return total_rows / self.total_duration

        return 0.0

    def to_dict(self, include_preds: bool = False) -> dict[str, Any]:
        """
        Convert the ModelConfiguration to a dictionary for serialization.

        Parameters
        ----------
        include_preds : bool, default False
            If True, includes pandas Series/DataFrames converted to native types.
            Set to False for standard JSON or Web compatibility.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the model configuration.
        """
        # Primary attributes and enums
        data: dict[str, Any] = {
            "id": self.id,
            "model_type": self.model_type.name,
            "task_type": self.task_type.name,
            "balancing_strategy": self.balancing_strategy.name,
            "optimization_strategy": self.optimization_strategy.name,
            "scoring": self.scoring.name,
            "model_params": self.model_params.to_dict(),
            "cv": self.cv,
            "n_jobs": self.n_jobs,
            "n_iter": self.n_iter,
            "max_iter": self.max_iter,
            # Performance indicators
            "has_test_set_evaluation_scores": self.has_test_set_evaluation_scores,
            "score_cv": self.score_cv,
            "score_train": self.score_train,
            "score_val": self.score_val,
            "score_val_cleaned": self.score_val_cleaned,
            "score_test": self.score_test,
            # Detailed metrics
            "mae_train": self.mae_train,
            "mae_val": self.mae_val,
            "mae_test": self.mae_test,
            "mse_train": self.mse_train,
            "mse_val": self.mse_val,
            "mse_test": self.mse_test,
            "r2_train": self.r2_train,
            "r2_val": self.r2_val,
            "r2_val_cleaned": self.r2_val_cleaned,
            "r2_test": self.r2_test,
            "accuracy_train": self.accuracy_train,
            "accuracy_val": self.accuracy_val,
            "accuracy_val_cleaned": self.accuracy_val_cleaned,
            "accuracy_test": self.accuracy_test,
            "acceptable_gap": self.acceptable_gap,
            "large_gap": self.large_gap,
            # Telemetry and auditing
            "tuning_duration": self.tuning_duration,
            "fit_duration": self.fit_duration,
            "available_gb": self.available_gb,
            "used_gb": self.used_gb,
            "estimated_peak_gb": self.estimated_peak_gb,
            "actual_peak_gb": self.actual_peak_gb,
            "memory_risk_triggered": self.memory_risk_triggered,
            "sampling_factor": self.sampling_factor,
            "concurrent_workers": self.concurrent_workers,
            "model_multiplier": self.model_multiplier,
            "num_candidates": self.num_candidates,
            "filter_outliers": self.filter_outliers,
            "outlier_count": self.outlier_count,
            "efficiency_threshold": self.efficiency_threshold,
            # Distribution summaries
            "train_mean": self.train_mean,
            "train_std": self.train_std,
            "train_median": self.train_median,
            "train_skew": self.train_skew,
            "train_kurtosis": self.train_kurtosis,
            "val_mean": self.val_mean,
            "val_std": self.val_std,
            "val_median": self.val_median,
            "val_skew": self.val_skew,
            "val_kurtosis": self.val_kurtosis,
            "test_mean": self.test_mean,
            "test_std": self.test_std,
            "test_median": self.test_median,
            "test_skew": self.test_skew,
            "test_kurtosis": self.test_kurtosis,
            "mean_delta": self.mean_delta,
            "std_delta": self.std_delta,
            "quality_score": self.quality_score,
            "drift_index": self.drift_index,
            # Nested Analysis
            "feature_analysis": (
                self.feature_analysis.to_dict() if self.feature_analysis else None
            ),
        }

        # Conversion of pandas types to serializable primitives
        if include_preds:
            data["preds_val"] = (
                self.preds_val.tolist() if self.preds_val is not None else None
            )
            data["probs_val"] = (
                self.probs_val.to_dict(orient="list")
                if self.probs_val is not None
                else None
            )
            data["preds_test"] = (
                self.preds_test.tolist() if self.preds_test is not None else None
            )
            data["probs_test"] = (
                self.probs_test.to_dict(orient="list")
                if self.probs_test is not None
                else None
            )

        return data

    def get_top_features(self, n: int = 1) -> dict[str, Any]:
        """
        Return a flattened dictionary of the top 'n' feature importance entries.

        Parameters
        ----------
        n : int, default 1
            The number of top features to extract.

        Returns
        -------
        dict[str, Any]
            A dictionary containing Top_Feature, Importance, and Cum_Importance
            entries suffixed by their rank.
        """
        feature_data: dict[str, Any] = {}

        # Cache the column indices and dataframe for performance
        f_idx = self.feature_analysis.get_feature_column_index
        i_idx = self.feature_analysis.get_importance_column_index
        c_idx = self.feature_analysis.get_cumulative_importance_column_index
        df = self.feature_analysis.feature_importances

        # Determine total features available for boundary checking
        available_features = len(self.feature_analysis.features)

        def get_top_n_feature_dict(
            index: int,
            feature: str | None,
            importance: float,
            cumulative_importance: float,
        ) -> dict[str, Any]:
            """Helper to generate suffixed dictionary keys."""
            suffix = f"_{index + 1}"
            return {
                f"Top_Feature{suffix}": feature,
                f"Importance{suffix}": round(importance, prefs.score_format.precision),
                f"Cum_Importance{suffix}": round(
                    cumulative_importance, prefs.score_format.precision
                ),
            }

        current_cum = 0.0
        for i in range(n):
            if i < available_features:
                # Use iat for high-speed scalar access by integer position
                feat = cast(str, df.iat[i, f_idx])
                imp = cast(float, df.iat[i, i_idx])
                current_cum = cast(float, df.iat[i, c_idx])

                ld = get_top_n_feature_dict(i, feat, imp, current_cum)
            else:
                # Provide padded None/0.0 values if requested 'n' exceeds feature count
                ld = get_top_n_feature_dict(i, None, 0.0, current_cum)

            feature_data.update(ld)

        return feature_data

    @classmethod
    def empty(cls, model_params: T_Params) -> ModelConfiguration[T_Params]:
        """
        Create an empty ModelConfiguration instance for initialization.

        Returns
        -------
        ModelConfiguration
            An empty configuration with baseline default values.
        """
        return cls(
            id="00",
            model_type=ModelType.UNKNOWN,
            task_type=TaskType.UNKNOWN,
            balancing_strategy=BalancingStrategy.NONE,
            optimization_strategy=OptimizationStrategy.MANUAL,
            model_params=model_params,
            cv=0,
            scoring=ScoringMetric.R2,
            n_jobs=0,
            n_iter=0,
            max_iter=0,
            score_cv=None,
            score_train=None,
            score_val=0.0,
            mae_train=None,
            mae_val=None,
            mse_train=None,
            mse_val=None,
            r2_train=None,
            r2_val=None,
            acceptable_gap=prefs.acceptable_gap,
            large_gap=prefs.large_gap,
        )

    def __hash__(self) -> int:
        """Make the configuration hashable for use in sets or dictionaries."""
        return hash(
            (
                self.model_type,
                self.balancing_strategy,
                self.optimization_strategy,
                self.score_val,
            )
        )

    def __eq__(self, other: object) -> bool:
        """Compare configurations primarily by their validation score."""
        if not isinstance(other, ModelConfiguration):
            return NotImplemented
        return self.score_val == other.score_val

    def __lt__(self, other: object) -> bool:
        """
        Order configurations by validation score (descending leaderboard).

        Handles None values by treating them as lower than any numeric score.
        """
        if not isinstance(other, ModelConfiguration):
            return NotImplemented

        # Logic for null-safe comparison
        if self.score_val is None and other.score_val is None:
            return False
        if self.score_val is None:
            return True
        if other.score_val is None:
            return False

        return self.score_val < other.score_val

    def info(self) -> str:
        """
        Return a formatted summary of key configuration and performance metrics.
        """
        metric_label = self.scoring.value.upper()

        # Formatter initialization from global preferences
        enum_v_fmt = EnumFormat()
        enum_n_fmt = EnumFormat(use_value=False)
        gen_fmt = EnumFormat(fallback=ModelGeneralization.PENDING.value)

        data: list[tuple[str, str]] = [
            ("Model Type", enum_v_fmt.format_value(self.model_type)),
            (
                "Balancing Strategy",
                enum_n_fmt.format_value(self.balancing_strategy.name),
            ),
            (
                "Optimization Strategy",
                enum_n_fmt.format_value(self.optimization_strategy.name),
            ),
            ("Parameters", self.model_params.info()),
            ("-" * 15, "-" * 15),
            (f"CV {metric_label}", prefs.score_format.format_value(self.score_cv)),
            (
                f"Train {metric_label}",
                prefs.score_format.format_value(self.score_train),
            ),
            (f"Valid {metric_label}", prefs.score_format.format_value(self.score_val)),
            ("Generalization", gen_fmt.format_value(self.model_generalization)),
            ("-" * 15, "-" * 15),
            ("Memory Available", f"{prefs.gb_format.format_value(self.available_gb)}"),
            ("Memory Used", f"{prefs.gb_format.format_value(self.used_gb)}"),
            ("Memory Peak", f"{prefs.gb_format.format_value(self.actual_peak_gb)}"),
            ("Memory Risk", f"{self.memory_risk_triggered}"),
            ("-" * 15, "-" * 15),
            ("Hyperparameters", ""),
        ]

        # Add dictionary parameters to the data list
        data.extend([(str(k), str(v)) for k, v in self.model_params.to_dict().items()])

        # Add Regression-specific context if applicable
        if self.r2_val is not None:
            data.extend(
                [
                    ("-" * 15, "-" * 15),
                    (
                        "R2 (Train/Val)",
                        f"{prefs.score_format.format_value(self.r2_train)} / "
                        f"{prefs.score_format.format_value(self.r2_val)}",
                    ),
                    ("R2 Gap", _f_audit_gap(self.r2_gap, self.model_generalization)),
                    (
                        "MAE (Train/Val)",
                        f"{prefs.score_format.format_value(self.mae_train)} / "
                        f"{prefs.score_format.format_value(self.mae_val)}",
                    ),
                    ("MAE Gap", _f_audit_gap(self.mae_gap, self.model_generalization)),
                ]
            )

        # Top Signals Summary
        top_3 = self.feature_analysis.features[:3]
        data.append(("Top Signals", ", ".join(top_3) if top_3 else "N/A"))

        return format_label_value_pairs(data)

    def detailed_feature_report(self) -> str:
        """Access the high-resolution feature importance report."""
        return self.feature_analysis.info()


@dataclass
class ModelAuditorConfig:
    """
    Configuration for the ModelAuditor orchestrator.

    This class centralizes evaluation parameters, pruning logic, and error
    reporting formats used during a model audit cycle.
    """

    data_splits: DataSplits
    dataset_name: str
    models_to_run: list[ModelSpecification] = field(default_factory=list)
    task_type: TaskType = TaskType.CLASSIFICATION

    # Evaluation settings
    cv: int = 5
    n_iter: int = -1
    scoring: ScoringMetric = ScoringMetric.F1
    top_n_importance: int = 1

    # Pruning and Drift logic
    viable_score_gap: float = 0.05
    auto_increment_phase: bool = True
    drift_threshold: float = prefs.drift_threshold

    # Feature Metadata
    features: dict[str, FeatureMetadata] = field(default_factory=dict)

    # Anomaly / Error Reporting
    top_n_anomalies: int = 5
    anomaly_display_map: dict[str, str] = field(default_factory=dict)
    actual_value_fmt: Any | None = None
    predicted_value_fmt: Any | None = None
    abs_error_fmt: Any | None = None
    error_pct_fmt: Any | None = None
    anomaly_threshold: float = prefs.anomaly_threshold
    anomaly_risk_concentration_threshold: float = (
        prefs.anomaly_risk_concentration_threshold
    )

    # Model Performance Thresholds
    model_accuracy_limit: float = prefs.model_accuracy_limit
    model_acceptable_limit: float = prefs.model_acceptable_limit
    model_stability_limit: float = prefs.model_stability_limit
    model_efficiency_threshold: int = prefs.model_efficiency_threshold

    def __post_init__(self) -> None:
        """Apply default job count and propagate to models."""
        # Default n_jobs to 3 unless already set
        if not hasattr(self, "_n_jobs"):
            self.n_jobs = 3

    @property
    def n_jobs(self) -> int:
        """Resolved parallel job count for model training."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int) -> None:
        """Validate and propagate n_jobs to all configured models."""
        self._n_jobs = validate_n_jobs(value)
        for m in self.models_to_run:
            m.n_jobs = self._n_jobs

    @classmethod
    def from_dataset(
        cls,
        dataset: pd.DataFrame,
        original_row_count: int,
        target_column: str,
        dataset_name: str,
        cv: int,
        model_classes: Sequence[type[ModelSpecification]],
        model_params: dict[type[ModelSpecification], ModelParams] | None = None,
        balancing_strategies: list[BalancingStrategy] | None = None,
        test_size: float = 0.2,
        valid_size: float = 0.2,
        random_state: Optional[int] = 42,
        scale_features: bool = True,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
        task_type: TaskType = TaskType.CLASSIFICATION,
        features: dict[str, FeatureMetadata] | None = None,
        **kwargs: Any,
    ) -> ModelAuditorConfig:
        """
        Factory method to create a config and instantiate models from raw data.

        Parameters
        ----------
        dataset : pd.DataFrame
            The raw source data.
        model_classes : Sequence[type[ModelSpecification]]
            The list of model types to instantiate (e.g., [RandomForest, Lasso]).
        balancing_strategies : list[BalancingStrategy], optional
            Strategies to apply (NONE, OVERSAMPLED, etc.).
        """
        from dsr_feature_eng_ml.models.model_specification import ModelSpecification

        all_features = [col for col in dataset.columns if col != target_column]

        # 1. Create the unified DataSplits
        splits = DataSplits.from_data_source(
            src=dataset,
            features_to_include=all_features,
            target_column=target_column,
            test_size=test_size,
            valid_size=valid_size,
            original_row_count=original_row_count,
            random_state=random_state,
            scale_features=scale_features,
        )

        # 2. Setup defaults for mutable types
        m_params = model_params or {}
        b_strategies = balancing_strategies or [BalancingStrategy.NONE]
        feat_meta = features or {}
        instantiated_models = []

        # 3. Model Instantiation Loop
        for m_cls in model_classes:
            base_params = m_params.get(m_cls)

            # Ensure params are updated with global state
            if base_params:
                base_params = dataclasses.replace(
                    base_params,
                    random_state=random_state,
                )

            for strategy in b_strategies:
                model_instance = ModelSpecification.instantiate_model(
                    model_cls=m_cls,
                    strategy=strategy,
                    params=base_params,
                    cv=cv,
                    optimization_strategy=optimization_strategy,
                    task_type=task_type,
                    **kwargs,
                )
                instantiated_models.append(model_instance)

        return cls(
            data_splits=splits,
            dataset_name=dataset_name,
            models_to_run=instantiated_models,
            task_type=task_type,
            features=feat_meta,
            cv=cv,
            **kwargs,
        )
