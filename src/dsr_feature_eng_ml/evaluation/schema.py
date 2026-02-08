from __future__ import annotations
import pandas as pd
from pandas.api.extensions import ExtensionDtype
import numpy as np
import functools
import dataclasses
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Optional,
    Any,
    Generic,
    TypeVar,
    TYPE_CHECKING,
    cast,
    Sequence,
    Union,
    List,
    Set,
)
from dsr_feature_eng_ml.enums import (
    ModelType,
    BalancingStrategy,
    OptimizationStrategy,
    ScoringMetric,
    ModelGeneralization,
    TaskType,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from dsr_utils.formatting import (
    TextAlignment,
    NumericScale,
    DataScale,
    FormatConfig,
    CurrencyFormat,
    PercentageFormat,
    IntegerFormat,
    FloatFormat,
    ValueDescFormat,
    DateTimeFormat,
    DataFormat,
    StringFormat,
    EnumFormat,
    BoolFormat,
    format_text,
    format_label_value_pairs,
)
from dsr_feature_eng_ml.preferences import prefs
from dsr_feature_eng_ml.utils.memory import validate_n_jobs

if TYPE_CHECKING:
    from dsr_feature_eng_ml.models.model_specification import ModelParams
    from dsr_feature_eng_ml.models.model_specification import ModelSpecification

T_Params = TypeVar("T_Params", bound="ModelParams")


def _f_audit_gap(gap: Optional[float], status: ModelGeneralization) -> str:
    """Private domain-specific wrapper for the audit report."""
    # return f_val_desc(val=gap, desc=status.value, fmt=prefs.score_format)
    audit_gap_format = ValueDescFormat(
        precision=prefs.score_format.precision, description=status.value
    )
    return audit_gap_format.format_value(gap)


class DatasetFormatters:
    @property
    def dtype_float(self) -> Union[
        CurrencyFormat,
        PercentageFormat,
        IntegerFormat,
        FloatFormat,
        ValueDescFormat,
        DateTimeFormat,
        DataFormat,
    ]:
        return self._dtype_float

    @dtype_float.setter
    def dtype_float(
        self,
        val: Union[
            CurrencyFormat,
            PercentageFormat,
            IntegerFormat,
            FloatFormat,
            ValueDescFormat,
            DateTimeFormat,
            DataFormat,
        ],
    ) -> None:
        self._dtype_float = val

    @property
    def dtype_object(self) -> Union[EnumFormat, StringFormat]:
        return self._dtype_object

    @dtype_object.setter
    def dtype_object(self, val: Union[EnumFormat, StringFormat]) -> None:
        self._dtype_object = val

    @property
    def dtype_int(self) -> Union[IntegerFormat, ValueDescFormat]:
        return self._dtype_int

    @dtype_int.setter
    def dtype_int(self, val: Union[IntegerFormat, ValueDescFormat]) -> None:
        self._dtype_int = val

    @property
    def dtype_bool(self) -> BoolFormat:
        return self._dtype_bool

    @dtype_bool.setter
    def dtype_bool(self, val: BoolFormat) -> None:
        self._dtype_bool = val

    @property
    def dtype_datetime(self) -> DateTimeFormat:
        return self._dtype_datetime

    @dtype_datetime.setter
    def dtype_datetime(self, val: DateTimeFormat) -> None:
        self._dtype_datetime = val

    @property
    def dtype_timedelta(self) -> DateTimeFormat:
        return self._dtype_timedelta

    @dtype_timedelta.setter
    def dtype_timedelta(self, val: DateTimeFormat) -> None:
        self._dtype_timedelta = val

    @property
    def dtype_category(self) -> StringFormat:
        return self._dtype_category

    @dtype_category.setter
    def dtype_category(self, val: StringFormat) -> None:
        self._dtype_category = val

    def __init__(
        self,
        dtype_float: Union[
            CurrencyFormat,
            PercentageFormat,
            IntegerFormat,
            FloatFormat,
            ValueDescFormat,
            DateTimeFormat,
            DataFormat,
        ] = FloatFormat(precision=2),
        dtype_object: Union[EnumFormat, StringFormat] = StringFormat(),
        dtype_int: Union[IntegerFormat, ValueDescFormat] = IntegerFormat(),
        dtype_bool: BoolFormat = BoolFormat(),
        dtype_datetime: DateTimeFormat = DateTimeFormat(
            date_format="%m-%d-%Y", time_format="%H:%M:%S"
        ),
        dtype_timedelta: DateTimeFormat = DateTimeFormat(use_duration_format=True),
        dtype_category: StringFormat = StringFormat(),
    ):
        self._dtype_float = dtype_float
        self._dtype_object = dtype_object
        self._dtype_int = dtype_int
        self._dtype_bool = dtype_bool
        self._dtype_datetime = dtype_datetime
        self._dtype_timedelta = dtype_timedelta
        self._dtype_category = dtype_category

    def fmt_for_dtype(
        self, input_dtype: Union[np.dtype[Any], ExtensionDtype]
    ) -> FormatConfig:
        # Handle ExtensionDtype first (CategoricalDtype, etc.)
        if isinstance(input_dtype, pd.CategoricalDtype):
            return self.dtype_category
        elif pd.api.types.is_datetime64_any_dtype(input_dtype):
            return self.dtype_datetime
        elif pd.api.types.is_timedelta64_dtype(input_dtype):
            return self.dtype_timedelta
        # np.isdtype() only works with NumPy dtypes, not ExtensionDtype
        elif isinstance(input_dtype, np.dtype):
            if np.isdtype(input_dtype, "integral"):
                return self.dtype_int
            elif np.isdtype(input_dtype, "real floating"):
                return self.dtype_float
            elif np.isdtype(input_dtype, "bool"):
                return self.dtype_bool
        return self.dtype_object


class FeatureMetadata:
    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> str:
        return self._id

    @property
    def position(self) -> int:
        return self._position

    @property
    def short_name(self) -> str:
        return self._short_name

    @short_name.setter
    def short_name(self, val: str) -> None:
        self._short_name = val

    @property
    def formatter(self) -> FormatConfig:
        return self._formatter

    @formatter.setter
    def formatter(self, val: FormatConfig) -> None:
        self._formatter = val

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, val: str) -> None:
        self._description = val

    @property
    def is_used_in_fit(self) -> bool:
        return self._is_used_in_fit

    @is_used_in_fit.setter
    def is_used_in_fit(self, val: bool) -> None:
        self._is_used_in_fit = val

    @property
    def parent_name(self) -> Optional[str]:
        return self._parent_name

    @parent_name.setter
    def parent_name(self, val: str) -> None:
        self._parent_name = val

    def __init__(
        self,
        name: str,  # Raw column name
        id: str,  # Feature ID
        position: int,  # Original index in dataset
        short_name: Optional[str],  # Clean name for charts
        formatter: FormatConfig = StringFormat(),  # Format
        description: str = "",  # User-provided context
        is_used_in_fit: bool = True,  # Inclusion status
        parent_name: Optional[str] = (
            None  # Name of the feature that should represent this feature in reports
        ),
    ):
        self._name = name
        self._id = id
        self._position = position
        self._short_name = short_name if short_name is not None else name
        self._formatter = formatter
        self._description = description
        self._is_used_in_fit = is_used_in_fit
        self._parent_name = parent_name

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "id": self.id,
            "position": self.position,
            "short_name": self.short_name,
            "description": self.description,
            "formatter": self.formatter.to_dict(),
            "is_used_in_fit": self.is_used_in_fit,
            "parent_name": self.parent_name,
        }

        return data

    @classmethod
    def dict_to_set(
        cls, feature_dict: dict[str, FeatureMetadata], target_column: str
    ) -> Set[FeatureMetadata]:
        feature_list: List[FeatureMetadata] = list(feature_dict.values())
        return cls.list_to_set(feature_list=feature_list, target_column=target_column)

    @classmethod
    def list_to_set(
        cls, feature_list: List[FeatureMetadata], target_column: str
    ) -> Set[FeatureMetadata]:
        return set(
            [f for f in feature_list if f.is_used_in_fit and f.name != target_column]
        )

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        formatters: DatasetFormatters,
        format_exceptions: dict[
            str,
            FormatConfig,
        ],
        feature_parent: dict[str, str] = {},
        exclude_from_fit: set[str] = set(),
        short_names: dict[str, str] = {},
    ) -> dict[str, FeatureMetadata]:
        fm_dict: dict[str, FeatureMetadata] = {}
        i = 0
        feature_count = len(df.columns)
        padding = max(len(str(feature_count)), 2)
        all_column_names = set(df.columns)
        id_fmt = IntegerFormat(width=padding, pad_value="0")

        for col in df.columns:
            if col in format_exceptions:
                fmt = format_exceptions[col]
            else:
                fmt = formatters.fmt_for_dtype(df[col].dtype)

            feature_id = f"F{id_fmt.format_value(i+1)}"
            parent_name = feature_parent.get(col)

            if parent_name and parent_name not in all_column_names:
                print(
                    f"WARNING: Feature '{col}' has invalid parent_name '{parent_name}'. Setting to None."
                )
                parent_name = None

            is_used_in_fit = col not in exclude_from_fit

            short_name = short_names.get(col, col)
            fm_dict[col] = FeatureMetadata(
                name=col,
                id=feature_id,
                position=i,
                formatter=fmt,
                short_name=short_name,
                is_used_in_fit=is_used_in_fit,
                parent_name=parent_name,
            )
            i += 1

        return fm_dict


class ModelFeatureImportance:
    """Analyzes and manages feature importance from tree-based models.

    Calculates feature importance scores, cumulative importance, and identifies
    optimal feature subsets based on importance thresholds.

    Attributes:
        features: Feature names sorted by importance (descending).
        feature_importances: DataFrame with columns 'feature', 'importance',
            and 'cumulative_importance'.
        threshold_80_idx: Index of first feature reaching 80% cumulative importance.
            Initially 0, set by calc_threshold_indices() or manually adjusted.
        threshold_95_idx: Index of first feature reaching 95% cumulative importance.
            Initially 0, set by calc_threshold_indices() or manually adjusted.

    Example:
        >>> importance = ModelFeatureImportance(
        ...     features=['age', 'income', 'tenure'],
        ...     model=trained_random_forest
        ... )
        >>> # Calculate threshold-based feature counts
        >>> importance.calc_threshold_indices()
        >>> print(f"80% threshold at index: {importance.threshold_80_idx}")
        >>>
        >>> # Or manually set for experimentation
        >>> importance.threshold_80_idx = 4
        >>> # Loop from 80% to 95% threshold indices
        >>> for n in range(importance.threshold_80_idx, importance.threshold_95_idx + 1):
        ...     top_features = importance.features[:n]

    Note:
        threshold_80_idx and threshold_95_idx should be within the valid index
        range of feature_importances (0 to len(features)). These values can be
        set automatically via calc_threshold_indices() or manually adjusted for
        experimentation with different feature subset sizes.
    """

    def __init__(
        self,
        feature_set: set[FeatureMetadata],
        importances: np.ndarray,
    ):
        # Create the dataframe using the array
        # feature_count = len(features)
        # id_fmt = f"0{max(2, len(str(feature_count)))}d"
        # feature_ids: list[str] = [f"F{i+1:{id_fmt}}" for i in range(feature_count)]
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

        # Convert importance and cumulative_importance to float32
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
        idx = self.feature_importances.columns.get_indexer_for(["feature"])
        if idx.size != 1 or idx[0] == -1:
            raise ValueError("Expected exactly one 'feature' column.")
        return int(idx[0])

    @property
    def get_importance_column_index(self) -> int:
        idx = self.feature_importances.columns.get_indexer_for(["importance"])
        if idx.size != 1 or idx[0] == -1:
            raise ValueError("Expected exactly one 'importance' column.")
        return int(idx[0])

    @property
    def get_cumulative_importance_column_index(self) -> int:
        idx = self.feature_importances.columns.get_indexer_for(
            ["cumulative_importance"]
        )
        if idx.size != 1 or idx[0] == -1:
            raise ValueError("Expected exactly one 'cumulative_importance' column.")
        return int(idx[0])

    @classmethod
    def empty(cls) -> ModelFeatureImportance:
        return cls(feature_set=set(), importances=np.array([]))

    def to_dict(self, include_full_df: bool = True) -> dict[str, Any]:
        """
        Converts feature importance data to a dictionary for serialization.

        Args:
            include_full_df: If True, converts the internal DataFrame to a list of dicts.
                            If False, only includes the high-level summary.
        """
        data = {
            "features": self.features,
            "threshold_80_idx": self.threshold_80_idx,
            "threshold_95_idx": self.threshold_95_idx,
        }

        if include_full_df:
            # Convert DataFrame to a list of records:
            # [{'feature': 'age', 'importance': 0.5, 'cumulative_importance': 0.5}, ...]
            data["feature_importances"] = self.feature_importances.to_dict(
                orient="records"
            )

        return data

    def info(self) -> str:
        """Display formatted feature importance information.

        Prints each feature with its importance score and cumulative importance
        percentage.
        """
        retval: str = ""

        for i in range(len(self.feature_importances)):
            feature = self.feature_importances.iloc[i]["feature"]
            importance = self.feature_importances.iloc[i]["importance"]
            cumulative_importance = self.feature_importances.iloc[i][
                "cumulative_importance"
            ]
            retval += "{:<3} {:<20} Importance: {:.4f}   {:>8.2%}\n".format(
                i + 1, feature, importance, cumulative_importance
            )

        return retval

    def calc_threshold_indices(self) -> None:
        """Calculate index positions for 80% and 95% cumulative importance thresholds.

        Determines the index of the first feature that reaches or exceeds 80%
        (threshold_80_idx) and 95% (threshold_95_idx) cumulative importance.
        These indices define a range for testing different feature subset sizes.

        Note:
            Sets the threshold_80_idx and threshold_95_idx attributes based on
            cumulative importance thresholds.
        """
        self.threshold_80_idx = 0
        self.threshold_95_idx = 0
        feature_count = len(self.feature_importances)

        for n in range(1, feature_count + 1):
            cumulative_importance = self.feature_importances.iloc[n - 1][
                "cumulative_importance"
            ]

            if self.threshold_80_idx == 0 and cumulative_importance >= 0.8:
                self.threshold_80_idx = n

            if self.threshold_95_idx == 0 and cumulative_importance >= 0.95:
                self.threshold_95_idx = n + 1
                break

        if self.threshold_95_idx > feature_count:
            self.threshold_95_idx = feature_count


@dataclass(frozen=True)
class DataSplits:
    """Immutable container for train/validation/test data splits in  workflows.

    This dataclass encapsulates dataset splits and provides factory methods for creating
    new instances with balanced training data through upsampling or downsampling. All
    instances are immutable - balancing operations return new instances rather than
    modifying existing ones.

    Attributes:
        features_to_include: Column names of features to include in the dataset.
        target_column: Name of the target variable column.
        test_features: Test set features for final model evaluation.
        test_target: Test set target values for final model evaluation.
        train_features: Training set features for model fitting.
        train_target: Training set target values for model fitting.
        val_features: Validation set features for hyperparameter tuning.
        val_target: Validation set target values for hyperparameter tuning.
        random_state: Random seed for reproducible operations.

    Example:
        >>> # Create initial splits from DataFrame
        >>> splits = DataSplits.from_data_source(
        ...     src=customer_df,
        ...     features_to_include=['age', 'income', 'tenure'],
        ...     target_column='churned',
        ...     test_size=0.2,
        ...     valid_size=0.25,
        ...     random_state=42
        ... )
        >>>
        >>> # Create balanced version (returns new instance)
        >>> balanced_splits = splits.with_upsampled_training()
        >>>
        >>> # Original splits unchanged, use balanced version for training
        >>> model.fit(balanced_splits.train_features, balanced_splits.train_target)

    Note:
        This is an immutable dataclass. Methods like with_upsampled_training() and
        with_downsampled_training() return new instances with modified training data
        rather than modifying the original instance in-place.
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
    random_state: int
    scaler: Optional[StandardScaler] = None

    @property
    def evaluation_features(self) -> list[str]:
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
        random_state: int,
        scale_features: bool = True,
        shuffle: bool = True,
        stratify: bool = False,
    ):
        """Create DataSplits from a source DataFrame with automatic train/valid/test splitting.

        Args:
            src (pd.DataFrame): Source DataFrame containing features and target.
            features_to_include (list[str]): Column names to use as features.
            target_column (str): Name of the target variable column.
            test_size (float): Proportion of data for test set (0.0 to 1.0).
            valid_size (float): Proportion of main data for validation (0.0 to 1.0).
            random_state (int): Random seed for reproducible splits.
            scale_features (bool): Whether to apply StandardScaler to features (default: True).
                Features are scaled using training set statistics, then validation and test
                sets are transformed using the same scaler. The target variable is never scaled.
                Scaling is beneficial for gradient descent-based models (Logistic Regression,
                Neural Networks, SVM) and does not affect tree-based models.
            shuffle (bool): Whether to shuffle data before splitting (default: True).
            stratify (bool): Whether to use stratified splitting based on target column (default: False).
                If True, the target variable will be used for stratification to preserve class distribution.

        Returns:
            DataSplits: New instance with train/validation/test splits.

        Example:
            >>> splits = DataSplits.from_data_source(
            ...     src=customer_df,
            ...     features_to_include=['age', 'income', 'tenure'],
            ...     target_column='churned',
            ...     test_size=0.2,
            ...     valid_size=0.25,
            ...     random_state=42,
            ...     scale_features=True,
            ...     shuffle=True,
            ...     stratify=True
            ... )
        """
        target = src[target_column]
        features = src[features_to_include]

        # Determine stratify parameter for train_test_split
        stratify_param = target if stratify else None

        # Create main (for training and validation) and test sets
        main_features, test_features, main_target, test_target = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_param,
        )

        # Create training and validation sets with stratification if requested
        stratify_valid = main_target if stratify else None
        train_features, val_features, train_target, val_target = train_test_split(
            main_features,
            main_target,
            test_size=valid_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_valid,
        )

        scaler_to_store = None
        if scale_features:
            # Identify which columns are numeric and which are categorical/object
            numeric_cols = train_features.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            categorical_cols = train_features.select_dtypes(
                exclude=[np.number]
            ).columns.tolist()
            scaler = StandardScaler()

            # Fit/Transform only numeric features
            train_numeric_scaled = scaler.fit_transform(train_features[numeric_cols])
            val_numeric_scaled = scaler.transform(val_features[numeric_cols])
            test_numeric_scaled = scaler.transform(test_features[numeric_cols])

            # Convert scaled arrays back to DataFrames
            train_features_scaled = pd.DataFrame(
                train_numeric_scaled, columns=numeric_cols, index=train_features.index
            )
            val_features_scaled = pd.DataFrame(
                val_numeric_scaled, columns=numeric_cols, index=val_features.index
            )
            test_features_scaled = pd.DataFrame(
                test_numeric_scaled, columns=numeric_cols, index=test_features.index
            )

            # Re-attach the categorical columns (unscaled)
            if categorical_cols:
                train_features = pd.concat(
                    [train_features_scaled, train_features[categorical_cols]], axis=1
                )
                val_features = pd.concat(
                    [val_features_scaled, val_features[categorical_cols]], axis=1
                )
                test_features = pd.concat(
                    [test_features_scaled, test_features[categorical_cols]], axis=1
                )
            else:
                train_features = train_features_scaled
                val_features = val_features_scaled
                test_features = test_features_scaled

            # Ensure columns stay in original order
            train_features = train_features[features_to_include]
            val_features = val_features[features_to_include]
            test_features = test_features[features_to_include]
            scaler_to_store = scaler

        return cls(
            features_to_include=features_to_include,
            target_column=target_column,
            test_features=test_features,
            test_target=test_target,
            train_features=train_features,
            train_target=train_target,
            val_features=val_features,
            val_target=val_target,
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
        """Create a new DataSplits with a subset of features from an existing instance.

        Args:
            src (DataSplits): Source DataSplits instance to copy from.
            features_to_include (list[str]): Subset of feature columns to include.

        Returns:
            DataSplits: New instance with only the specified features.

        Example:
            >>> top_features = ['feature1', 'feature2', 'feature3']
            >>> reduced_splits = DataSplits.from_data_splits(
            ...     src=original_splits,
            ...     features_to_include=top_features
            ... )
        """
        return cls(
            features_to_include=features_to_include,
            target_column=src.target_column,
            test_features=pd.DataFrame(src.test_features[features_to_include].copy()),
            test_target=src.test_target.copy(),
            train_features=pd.DataFrame(src.train_features[features_to_include].copy()),
            train_target=src.train_target.copy(),
            val_features=pd.DataFrame(src.val_features[features_to_include].copy()),
            val_target=src.val_target.copy(),
            original_row_count=src.original_row_count,
            random_state=src.random_state,
            scaler=src.scaler,
        )

    @classmethod
    def empty(cls) -> DataSplits:
        """Create an empty DataSplits instance for initialization purposes.

        Returns:
            DataSplits: Empty instance with no data.
        """
        empty_df: pd.DataFrame = pd.DataFrame()
        empty_series: pd.Series = pd.Series(dtype=object)

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

    def info(self):
        """Display information about all data splits.

        Prints DataFrame info for main, test, train, and validation datasets.
        """
        self.test_features.info()
        self.train_features.info()
        self.val_features.info()

    def inverse_transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to revert a DataFrame to original units for display."""
        if self.scaler is None:
            return df.copy()

        df_inv = df.copy()
        numeric_cols = df_inv.select_dtypes(include=[np.number]).columns.tolist()

        # We only transform columns the scaler knows about
        scaled_feature_names = list(self.scaler.feature_names_in_)
        cols_to_revert = [c for c in numeric_cols if c in scaled_feature_names]

        df_inv[cols_to_revert] = self.scaler.inverse_transform(df_inv[cols_to_revert])
        return df_inv

    def with_upsampled_training(
        self, X: pd.DataFrame, y: pd.Series[Any], use_combined_data: bool = False
    ) -> DataSplits:
        """Return a new DataSplits instance with upsampled training data.

        Identifies the minority class and duplicates its samples to match the majority
        class size. The resulting balanced dataset is shuffled to mix the classes.

        Returns:
            DataSplits: New instance with balanced training data. Validation and
                test sets remain unchanged.

        Example:
            >>> upsampled_splits = data_splits.with_upsampled_training()
            >>> # New instance has balanced class distribution
        """
        # This version works with a True/False target
        features_zeros = X[y == 0]
        features_ones = X[y == 1]
        target_zeros = y[y == 0]
        target_ones = y[y == 1]
        N0 = len(target_zeros)
        N1 = len(target_ones)

        if N0 > N1:
            # Minority class is 1 (N1)
            features_min, features_maj = features_ones, features_zeros
            target_min, target_maj = target_ones, target_zeros
        else:
            # Minority class is 0 (N0)
            features_min, features_maj = features_zeros, features_ones
            target_min, target_maj = target_zeros, target_ones

        increase_factor = int(len(features_maj) / len(features_min))
        features_upsampled = pd.concat(
            [features_maj] + [features_min] * increase_factor
        )
        target_upsampled = pd.concat([target_maj] + [target_min] * increase_factor)

        # Shuffle and explicitly convert back to DataFrame/Series
        shuffled_features, shuffled_target = shuffle(  # type: ignore[misc]
            features_upsampled, target_upsampled, random_state=self.random_state
        )

        # Explicitly convert to proper types for dataclass constructor
        train_features_upsampled = pd.DataFrame(
            shuffled_features,  # type: ignore[arg-type]
            columns=features_upsampled.columns,
        )

        train_target_upsampled = pd.Series(
            shuffled_target, name=target_upsampled.name  # type: ignore[arg-type]
        )

        return DataSplits(
            features_to_include=self.features_to_include,
            target_column=self.target_column,
            test_features=self.test_features,
            test_target=self.test_target,
            train_features=train_features_upsampled,
            train_target=train_target_upsampled,
            val_features=pd.DataFrame() if use_combined_data else self.val_features,
            val_target=pd.Series() if use_combined_data else self.val_target,
            original_row_count=self.original_row_count,
            random_state=self.random_state,
            scaler=self.scaler,
        )

    def with_downsampled_training(
        self, X: pd.DataFrame, y: pd.Series[Any], use_combined_data: bool = False
    ) -> DataSplits:
        """Return a new DataSplits instance with downsampled training data.

        Identifies the majority class and randoy samples from it to match the minority
        class size. The resulting balanced dataset is shuffled to mix the classes.

        Returns:
            DataSplits: New instance with balanced training data. Validation and
                test sets remain unchanged. Some majority class samples are discarded.

        Example:
            >>> downsampled_splits = data_splits.with_downsampled_training()
            >>> # New instance has balanced class distribution with fewer samples
        """
        # This version works with a True/False target
        features_zeros = X[y == 0]
        features_ones = X[y == 1]
        target_zeros = y[y == 0]
        target_ones = y[y == 1]
        N0 = len(target_zeros)
        N1 = len(target_ones)

        if N0 > N1:
            # Minority class is 1 (N1)
            features_min, features_maj = features_ones, features_zeros
            target_min, target_maj = target_ones, target_zeros
        else:
            # Minority class is 0 (N0)
            features_min, features_maj = features_zeros, features_ones
            target_min, target_maj = target_zeros, target_ones

        decrease_factor = len(features_min) / len(features_maj)
        features_downsampled = pd.concat(
            [features_min]
            + [
                features_maj.sample(
                    frac=decrease_factor, random_state=self.random_state
                )
            ]
        )
        target_downsampled = pd.concat(
            [target_min]
            + [target_maj.sample(frac=decrease_factor, random_state=self.random_state)]
        )

        shuffled_features, shuffled_target = shuffle(  # type: ignore[misc]
            features_downsampled, target_downsampled, random_state=self.random_state
        )

        # Explicitly convert to proper types for dataclass constructor
        train_features_downsampled = pd.DataFrame(
            shuffled_features,  # type: ignore[arg-type]
            columns=features_downsampled.columns,
        )

        train_target_downsampled = pd.Series(
            shuffled_target, name=target_downsampled.name  # type: ignore[arg-type]
        )

        return DataSplits(
            features_to_include=self.features_to_include,
            target_column=self.target_column,
            test_features=self.test_features,
            test_target=self.test_target,
            train_features=train_features_downsampled,
            train_target=train_target_downsampled,
            val_features=pd.DataFrame() if use_combined_data else self.val_features,
            val_target=pd.Series() if use_combined_data else self.val_target,
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
        Returns training features and target based on the requested strategy.
        """
        if use_combined_data:
            feature_list = [f.name for f in feature_set]
            X = pd.concat(
                [
                    self.train_features[feature_list],
                    self.val_features[feature_list],
                ]
            )
            y = pd.concat([self.train_target, self.val_target])
        else:
            X = self.train_features
            y = self.val_target

        if strategy == BalancingStrategy.OVERSAMPLED:
            balanced_splits = self.with_upsampled_training(X, y, use_combined_data)
            return balanced_splits.train_features, balanced_splits.train_target

        elif strategy == BalancingStrategy.UNDERSAMPLED:
            balanced_splits = self.with_downsampled_training(X, y, use_combined_data)
            return balanced_splits.train_features, balanced_splits.train_target

        # Default: No balancing (BalancingStrategy.NONE)
        return X, y

    def get_train_weights(
        self, strategy: BalancingStrategy, is_regression: bool = True
    ) -> Optional[np.ndarray]:
        """
        Returns an array of sample weights for the training data.
        """
        if strategy != BalancingStrategy.WEIGHTED:
            return None

        y = self.train_target

        if is_regression:
            # Use histogram binning to weight rare fare values
            counts, bin_edges = np.histogram(y, bins=20)
            # SMOOTHING: Add a constant (k) to prevent extreme weights for tiny bins
            # A good k is often the mean count per bin
            k = len(y) / 20
            smoothed_counts = counts + k
            # Map each sample to its bin's inverse frequency
            bin_indices = np.digitize(y, bin_edges[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
            weights = 1.0 / smoothed_counts[bin_indices]
        else:
            # For classification, use inverse class frequency
            class_counts = y.value_counts()
            weights = y.map(1.0 / class_counts)

        # Normalize weights so the mean is 1.0 (improves stability)
        normalized_weights = weights / weights.mean()

        if isinstance(normalized_weights, pd.Series):
            return cast(np.ndarray, normalized_weights.to_numpy())

        return cast(np.ndarray, normalized_weights)

    def with_feature_subset(self, feature_subset: list[str]) -> DataSplits:
        """Create new DataSplits with only specified features, reusing target data.

        Memory-efficient factory method that creates a new DataSplits instance
        with a subset of features. Reuses the existing target Series objects
        (which don't change) and creates new DataFrames only for the selected
        feature columns. This avoids unnecessary duplication of target data.

        Args:
            feature_subset (list[str]): List of feature column names to include.

        Returns:
            DataSplits: New instance with only the specified features.
                Target Series are shared (not copied) for memory efficiency.

        Example:
            >>> # Original splits with 10 features
            >>> original_splits = DataSplits.from_data_source(...)
            >>> # Create new splits with only top 5 features
            >>> top_5_splits = original_splits.with_feature_subset(top_5_features)

        Note:
            This method creates new DataFrame objects for features but reuses
            the target Series, significantly reducing memory usage compared to
            creating entirely new DataSplits from scratch.
        """
        return DataSplits(
            features_to_include=feature_subset,
            target_column=self.target_column,
            test_features=pd.DataFrame(self.test_features[feature_subset]),
            test_target=self.test_target,  # Reuse same Series
            train_features=pd.DataFrame(self.train_features[feature_subset]),
            train_target=self.train_target,  # Reuse same Series
            val_features=pd.DataFrame(self.val_features[feature_subset]),
            val_target=self.val_target,  # Reuse same Series
            original_row_count=self.original_row_count,
            random_state=self.random_state,
            scaler=self.scaler,
        )

    def auc_roc_curve(
        self, test_proba: np.ndarray, plot_title: str = "ROC Curve"
    ) -> float:
        """Plot ROC curve and calculate AUC score for model predictions.

        Generates a Receiver Operating Characteristic (ROC) curve showing the
        trade-off between true positive rate and false positive rate. Calculates
        and displays the Area Under the Curve (AUC) score.

        Args:
            test_proba (np.ndarray): Predicted probabilities for positive class on test set.
            plot_title (str): Title for the ROC curve plot.

        Returns:
            float: AUC (Area Under the Curve) score, ranging from 0 to 1.
                   Higher values indicate better model performance.

        Note:
            AUC of 0.5 indicates random guessing (diagonal line).
            AUC of 1.0 indicates perfect classification.
        """
        from sklearn.metrics import auc

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

        # ROC curve for random model (looks like a straight line)
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier"
        )

        # Annotate threshold points
        threshold_format = FloatFormat(precision=2)
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

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title(plot_title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()

        return auc_score


@dataclass
class ModelConfigurationStats:
    @dataclass
    class ModelSplitStats:
        class SplitType(Enum):
            TRAIN_VAL = auto()
            TEST = auto()

        mean: float
        std: float
        median: float
        skew: float
        kurtosis: float

    model_split_stats: dict[str, ModelConfigurationStats.ModelSplitStats]

    # The following attributes are valid only for SplitType.TRAIN_VAL:
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
    ) -> "ModelConfigurationStats":
        def get_scalar(val):
            return float(pd.Series(val).iloc[0])

        match split_type:
            case ModelConfigurationStats.ModelSplitStats.SplitType.TRAIN_VAL:
                train_target_array = data_splits.train_target.to_numpy()
                train_split_stats = ModelConfigurationStats.ModelSplitStats(
                    mean=get_scalar(data_splits.train_target.mean()),
                    std=get_scalar(data_splits.train_target.std()),
                    median=get_scalar(data_splits.train_target.median()),
                    skew=float(skew(train_target_array)),
                    kurtosis=float(kurtosis(train_target_array)),
                )
                val_target_array = data_splits.val_target.to_numpy()
                val_split_stats = ModelConfigurationStats.ModelSplitStats(
                    mean=get_scalar(data_splits.val_target.mean()),
                    std=get_scalar(data_splits.val_target.std()),
                    median=get_scalar(data_splits.val_target.median()),
                    skew=float(skew(val_target_array)),
                    kurtosis=float(kurtosis(val_target_array)),
                )

                # Scoring Logic
                cleaned_score = (
                    config.score_val_cleaned
                    if config.score_val_cleaned is not None
                    else 0.0
                )
                raw_score = config.score_val if config.score_val is not None else 0.0

                if cleaned_score <= raw_score:
                    quality_score = 100.0
                else:
                    gap = cleaned_score - raw_score
                    penalty_multiplier = prefs.get_penalty_multiplier_for_task_type(
                        config.task_type
                    )
                    quality_score = 100.0 - (gap * penalty_multiplier)

                # Delta Logic
                t_mean = train_split_stats.mean
                v_mean = val_split_stats.mean
                t_std = train_split_stats.std
                v_std = val_split_stats.std
                drift_index = abs(t_mean - v_mean) / t_mean if t_mean != 0 else 0.0
                score_cv = config.score_cv if config.score_cv is not None else 0.0
                mean_delta = raw_score - score_cv
                std_delta = (abs(t_std - v_std) / t_std * 100) if t_std != 0 else 0.0
                return cls(
                    model_split_stats={
                        "train": train_split_stats,
                        "val": val_split_stats,
                    },
                    quality_score=quality_score,
                    drift_index=drift_index,
                    mean_delta=mean_delta,
                    std_delta=std_delta,
                )
            case ModelConfigurationStats.ModelSplitStats.SplitType.TEST:
                test_target_array = data_splits.test_target.to_numpy()
                test_split_stats = ModelConfigurationStats.ModelSplitStats(
                    mean=get_scalar(data_splits.test_target.mean()),
                    std=get_scalar(data_splits.test_target.std()),
                    median=get_scalar(data_splits.test_target.median()),
                    skew=float(skew(test_target_array)),
                    kurtosis=float(kurtosis(test_target_array)),
                )
                return cls(
                    model_split_stats={
                        "test": test_split_stats,
                    },
                    quality_score=0.0,
                    drift_index=0.0,
                    mean_delta=0.0,
                    std_delta=0.0,
                )


@dataclass(frozen=True)
@functools.total_ordering
class ModelConfiguration(Generic[T_Params]):
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
    score_cv: Optional[float] = None
    score_train: Optional[float] = None
    score_val: Optional[float] = None
    score_val_cleaned: Optional[float] = None
    score_test: Optional[float] = None
    mae_train: Optional[float] = None
    mae_val: Optional[float] = None
    mae_test: Optional[float] = None
    mse_train: Optional[float] = None
    mse_val: Optional[float] = None
    mse_test: Optional[float] = None
    r2_train: Optional[float] = None
    r2_val: Optional[float] = None
    r2_val_cleaned: Optional[float] = None
    r2_test: Optional[float] = None
    accuracy_train: Optional[float] = None
    accuracy_val: Optional[float] = None
    accuracy_val_cleaned: Optional[float] = None
    accuracy_test: Optional[float] = None
    preds_val: Optional[pd.Series] = None
    probs_val: Optional[pd.DataFrame] = None
    preds_test: Optional[pd.Series] = None
    probs_test: Optional[pd.DataFrame] = None
    acceptable_gap: float = prefs.acceptable_gap
    large_gap: float = prefs.large_gap
    feature_analysis: ModelFeatureImportance = field(
        default_factory=lambda: ModelFeatureImportance.empty()
    )
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
    filter_outliers: bool = False
    outlier_count: int = prefs.default_worst_errors_n
    efficiency_threshold: int = 0
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
    mean_delta: float = 0.0
    std_delta: float = 0.0
    quality_score: float = 0.0
    drift_index: float = 0.0

    @property
    def r2_gap(self) -> float:
        if self.r2_train is None or self.r2_val is None:
            return 0.0
        return abs(self.r2_train - self.r2_val)

    @property
    def mae_gap(self) -> float:
        if self.mae_train is None or self.mae_val is None:
            return 0.0
        # Note: For error metrics, we often use (Val - Train)
        # so a positive number still means "Doing worse on Validation"
        return self.mae_val - self.mae_train

    @property
    def accuracy_gap(self) -> float:
        """
        Returns the absolute difference between train and validation accuracy.
        """
        if self.accuracy_train is None or self.accuracy_val is None:
            return 0.0
        return abs(self.accuracy_train - self.accuracy_val)

    @property
    def gap(self) -> float:
        """
        Returns the primary performance gap.
        """
        if self.task_type == TaskType.REGRESSION:
            return self.r2_gap
        elif self.task_type == TaskType.CLASSIFICATION:
            return self.accuracy_gap

        # Return 0.0 if unknown to avoid breaking the leaderboard display
        return 0.0

    @property
    def model_generalization(self) -> ModelGeneralization:
        # Use the primary scores (which could be F1 or R2 depending on task)
        if self.score_train is None or self.score_val is None:
            return ModelGeneralization.PENDING

        gap = self.gap

        if gap > self.large_gap:
            return ModelGeneralization.OVERFIT
        elif gap > self.acceptable_gap:
            return ModelGeneralization.MARGINAL
        return ModelGeneralization.WELL_FIT

    @property
    def params_dict(self) -> dict:
        """Helper for Scikit-Learn"""
        return self.model_params.to_dict()

    @property
    def total_duration(self) -> float:
        return self.tuning_duration + self.fit_duration

    @property
    def total_duration_min(self) -> float:
        return self.total_duration / 60.0

    @property
    def train_score(self) -> float:
        return self.score_train if self.score_train is not None else 0.0

    @property
    def val_score(self) -> float:
        return self.score_val if self.score_val is not None else 0.0

    @property
    def test_score(self) -> float:
        return self.score_test if self.score_test is not None else 0.0

    def efficiency(self, data_splits: DataSplits) -> float:
        if self.total_duration > 0.0:
            return (
                len(data_splits.train_features) + len(data_splits.val_features)
            ) / self.total_duration
        else:
            return 0.0

    def to_dict(self, include_preds: bool = False) -> dict[str, Any]:
        """
        Converts the ModelConfiguration to a dictionary for serialization.

        Args:
            include_preds: If True, includes pandas Series/DataFrames.
                          Set to False for JSON/Web compatibility.
        """
        data = {
            "id": self.id,
            "model_type": self.model_type.name,
            "task_type": self.task_type.name,
            "balancing_strategy": self.balancing_strategy.name,
            "optimization_strategy": self.optimization_strategy.name,
            "scoring": self.scoring.name,
            # Recursive to_dict calls for nested custom objects
            "model_params": self.model_params.to_dict(),
            "cv": self.cv,
            "n_jobs": self.n_jobs,
            "n_iter": self.n_iter,
            "max_iter": self.max_iter,
            # Performance metrics
            "has_test_set_evaluation_scores": self.has_test_set_evaluation_scores,
            "score_cv": self.score_cv,
            "score_train": self.score_train,
            "score_val": self.score_val,
            "score_val_cleaned": self.score_val_cleaned,
            "score_test": self.score_test,
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
            # Hardware/Duration stats
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
            # Statistics for Audit
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
            # Feature Analysis
            "feature_analysis": (
                self.feature_analysis.to_dict()
                if hasattr(self.feature_analysis, "to_dict")
                else None
            ),
        }

        # Handle Pandas objects conditionally (essential for JSON compatibility)
        if include_preds:
            data["preds_val"] = (
                self.preds_val.to_list() if self.preds_val is not None else None
            )
            data["probs_val"] = (
                self.probs_val.to_dict() if self.probs_val is not None else None
            )
            data["preds_test"] = (
                self.preds_test.to_list() if self.preds_test is not None else None
            )
            data["probs_test"] = (
                self.probs_test.to_dict() if self.probs_test is not None else None
            )

        return data

    def get_top_features(self, n: int = 1) -> dict:
        feature_data = {}

        # Cache the column indices
        f_idx = self.feature_analysis.get_feature_column_index
        i_idx = self.feature_analysis.get_importance_column_index
        c_idx = self.feature_analysis.get_cumulative_importance_column_index
        df = self.feature_analysis.feature_importances

        def get_top_n_feature_dict(
            index: int,
            feature: Optional[str],
            importance: float,
            cumulative_importance: float,
        ) -> dict:
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
            if i < len(self.feature_analysis.features):
                # Extract values once and cast for the type checker
                feat = cast(str, df.iat[i, f_idx])
                imp = cast(float, df.iat[i, i_idx])
                current_cum = cast(float, df.iat[i, c_idx])

                ld = get_top_n_feature_dict(i, feat, imp, current_cum)
            else:
                # Padding
                ld = get_top_n_feature_dict(i, None, 0.0, current_cum)

            # Merge the triplet dictionary into our main return dictionary
            feature_data.update(ld)

        return feature_data

    @classmethod
    def empty(cls, model_params: T_Params) -> ModelConfiguration[T_Params]:
        """Create an empty ModelConfiguration instance for initialization.

        Returns:
            ModelConfiguration: Empty configuration with default values.
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
        """Make configuration hashable for use in sets/dicts."""
        return hash(
            (
                self.model_type,
                self.balancing_strategy,
                self.optimization_strategy,
                self.score_val,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelConfiguration):
            return NotImplemented
        return self.score_val == other.score_val

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ModelConfiguration):
            return NotImplemented
        # Handle None values: None is considered "less than" any numeric value
        if self.score_val is None and other.score_val is None:
            return False
        if self.score_val is None:
            return True
        if other.score_val is None:
            return False
        return self.score_val < other.score_val

    def info(self) -> str:
        # 1. Primary Metric Labeling
        metric_label = self.scoring.value.upper()

        # 2. Build Data List
        enum_value_format = EnumFormat()
        enum_name_format = EnumFormat(use_value=False)
        generalization_format = EnumFormat(fallback=ModelGeneralization.PENDING.value)
        data = [
            ("Model Type", enum_value_format.format_value(self.model_type)),
            (
                "Balancing Strategy",
                enum_name_format.format_value(self.balancing_strategy.name),
            ),
            (
                "Optimization Strategy",
                enum_name_format.format_value(self.optimization_strategy.name),
            ),
            ("Parameters", self.model_params.info()),
            ("-" * 10, "-" * 10),
            (f"CV {metric_label}", prefs.score_format.format_value(self.score_cv)),
            (
                f"Train {metric_label}",
                prefs.score_format.format_value(self.score_train),
            ),
            (f"Valid {metric_label}", prefs.score_format.format_value(self.score_val)),
            (
                "Generalization",
                generalization_format.format_value(self.model_generalization),
            ),
            ("Memory Available", f"{prefs.gb_format.format_value(self.available_gb)}"),
            ("Memory Used", f"{prefs.gb_format.format_value(self.used_gb)}"),
            (
                "Memory Est Peak",
                f"{prefs.gb_format.format_value(self.estimated_peak_gb)}",
            ),
            ("Memory Peak", f"{prefs.gb_format.format_value(self.actual_peak_gb)}"),
            ("Memory Risk Triggered", f"{self.memory_risk_triggered}"),
            ("-" * 10, "-" * 10),
            ("Optimized Parameters", ""),
        ]

        data.extend(
            [
                (str(param), str(val))
                for param, val in self.model_params.to_dict().items()
            ]
        )

        # 3. Add Regression Slices if available
        if self.r2_val is not None:
            data.extend(
                [
                    ("-" * 10, "-" * 10),
                    (
                        "R2 (Train/Val)",
                        f"{prefs.score_format.format_value(self.r2_train)} / {prefs.score_format.format_value(self.r2_val)}",
                    ),
                    ("R2 Gap", _f_audit_gap(self.r2_gap, self.model_generalization)),
                    (
                        "MAE (Train/Val)",
                        f"{prefs.score_format.format_value(self.mae_train)} / {prefs.score_format.format_value(self.mae_val)}",
                    ),
                    ("MAE Gap", _f_audit_gap(self.mae_gap, self.model_generalization)),
                    (
                        "MSE (Train/Val)",
                        f"{prefs.score_format.format_value(self.mse_train)} / {prefs.score_format.format_value(self.mse_val)}",
                    ),
                ]
            )

        # Add a high-level summary to the main report
        top_3 = self.feature_analysis.features[:3]
        data.append(("Top Signals", ", ".join(top_3) if top_3 else "N/A"))

        return format_label_value_pairs(data)

    def detailed_feature_report(self) -> str:
        """Accesses the deep-dive report from the analysis object."""
        return self.feature_analysis.info()


@dataclass
class ModelAuditorConfig:
    """Configuration for the ModelAuditor orchestrator."""

    from dsr_utils.formatting import FormatType

    data_splits: DataSplits
    dataset_name: str
    models_to_run: list[ModelSpecification] = field(default_factory=list)
    task_type: TaskType = TaskType.CLASSIFICATION

    # Evaluation settings
    cv: int = 5
    n_iter: int = -1
    scoring: ScoringMetric = ScoringMetric.F1
    top_n_importance: int = 1

    # Logic for pruning models that underperform
    viable_score_gap: float = 0.05
    auto_increment_phase: bool = True
    drift_threshold: float = prefs.drift_threshold

    # Features
    features: dict[str, FeatureMetadata] = field(default_factory=dict)

    # Anomalies / Errors
    top_n_anomalies: int = 5
    anomaly_display_map: dict = field(default_factory=dict)
    actual_value_fmt: Any = None
    predicted_value_fmt: Any = None
    abs_error_fmt: Any = None
    error_pct_fmt: Any = None
    anomaly_threshold: float = prefs.anomaly_threshold
    anomaly_risk_concentration_threshold: float = (
        prefs.anomaly_risk_concentration_threshold
    )

    # Model Thresholds
    model_accuracy_limit: float = prefs.model_accuracy_limit
    model_acceptable_limit: float = prefs.model_acceptable_limit
    model_stability_limit: float = prefs.model_stability_limit
    model_efficiency_threshold: int = prefs.model_efficiency_threshold

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: int):
        self._n_jobs = validate_n_jobs(value)

        for m in self.models_to_run:
            m.n_jobs = self.n_jobs

    def __post_init__(self):
        self.n_jobs = 3

    @classmethod
    def from_dataset(
        cls,
        dataset: pd.DataFrame,
        original_row_count: int,
        target_column: str,
        dataset_name: str,
        cv: int,
        model_classes: Sequence[type[ModelSpecification]],
        model_params: Optional[dict[type[ModelSpecification], ModelParams]] = None,
        balancing_strategies: list[BalancingStrategy] = field(
            default_factory=lambda: [BalancingStrategy.NONE]
        ),
        test_size: float = 0.2,
        valid_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
        task_type: TaskType = TaskType.CLASSIFICATION,
        features: dict[str, FeatureMetadata] = field(default_factory=dict),
        **kwargs,
    ) -> ModelAuditorConfig:
        """Factory method to create a config and instantiate models."""
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

        # 2. List Comprehension:
        model_params = model_params or {}
        instantiated_models = []

        for m_cls in model_classes:
            params = model_params.get(m_cls)

            if params:
                params = dataclasses.replace(
                    params,
                    random_state=random_state,
                    optimization_strategy=optimization_strategy,
                )

            for strategy in balancing_strategies:
                model_instance = ModelSpecification.instantiate_model(
                    model_cls=m_cls,
                    strategy=strategy,
                    params=params,
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
            features=features,
            **kwargs,
        )
