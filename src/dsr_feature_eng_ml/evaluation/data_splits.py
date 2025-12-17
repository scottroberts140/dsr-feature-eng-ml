from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


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
        features_main: Combined training and validation features (before split).
        target_main: Combined training and validation target values (before split).
        features_test: Test set features for final model evaluation.
        target_test: Test set target values for final model evaluation.
        features_train: Training set features for model fitting.
        target_train: Training set target values for model fitting.
        features_valid: Validation set features for hyperparameter tuning.
        target_valid: Validation set target values for hyperparameter tuning.
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
        >>> model.fit(balanced_splits.features_train, balanced_splits.target_train)

    Note:
        This is an immutable dataclass. Methods like with_upsampled_training() and
        with_downsampled_training() return new instances with modified training data
        rather than modifying the original instance in-place.
    """
    features_to_include: list[str]
    target_column: str
    features_main: pd.DataFrame
    target_main: pd.Series
    features_test: pd.DataFrame
    target_test: pd.Series
    features_train: pd.DataFrame
    target_train: pd.Series
    features_valid: pd.DataFrame
    target_valid: pd.Series
    random_state: int

    @classmethod
    def from_data_source(
            cls,
            src: pd.DataFrame,
            features_to_include: list[str],
            target_column: str,
            test_size: float,
            valid_size: float,
            random_state: int,
            scale_features: bool = True,
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
            ...     scale_features=True
            ... )
        """
        target = src[target_column]
        features = src[features_to_include]
        test_size = test_size
        valid_size = valid_size
        random_state = random_state

        # Create main (for training and validation) and test sets
        features_main, features_test, target_main, target_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state
        )

        # Create training and validation sets
        features_train, features_valid, target_train, target_valid = train_test_split(
            features_main,
            target_main,
            test_size=valid_size,
            random_state=random_state
        )

        if scale_features:
            scaler = StandardScaler()

            # Fit on training features only
            features_train_scaled = scaler.fit_transform(features_train)

            # Transform validation/test using training statistics
            features_valid_scaled = scaler.transform(features_valid)
            features_test_scaled = scaler.transform(features_test)

            # Convert back to DataFrames to preserve column and index information
            features_train = pd.DataFrame(
                features_train_scaled, columns=features_train.columns, index=features_train.index)
            features_valid = pd.DataFrame(
                features_valid_scaled, columns=features_valid.columns, index=features_valid.index)
            features_test = pd.DataFrame(
                features_test_scaled, columns=features_test.columns, index=features_test.index)

        return cls(
            features_to_include=features_to_include,
            target_column=target_column,
            features_main=features_main,
            target_main=target_main,
            features_test=features_test,
            target_test=target_test,
            features_train=features_train,
            target_train=target_train,
            features_valid=features_valid,
            target_valid=target_valid,
            random_state=random_state
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
            features_main=pd.DataFrame(
                src.features_main[features_to_include].copy()),
            target_main=src.target_main.copy(),
            features_test=pd.DataFrame(
                src.features_test[features_to_include].copy()),
            target_test=src.target_test.copy(),
            features_train=pd.DataFrame(
                src.features_train[features_to_include].copy()),
            target_train=src.target_train.copy(),
            features_valid=pd.DataFrame(
                src.features_valid[features_to_include].copy()),
            target_valid=src.target_valid.copy(),
            random_state=src.random_state,
        )

    @classmethod
    def empty(
        cls
    ) -> DataSplits:
        """Create an empty DataSplits instance for initialization purposes.

        Returns:
            DataSplits: Empty instance with no data.
        """
        empty_df: pd.DataFrame = pd.DataFrame()
        empty_series: pd.Series = pd.Series(dtype=object)

        return cls(
            features_to_include=[],
            target_column='',
            features_main=empty_df,
            target_main=empty_series,
            features_test=empty_df,
            target_test=empty_series,
            features_train=empty_df,
            target_train=empty_series,
            features_valid=empty_df,
            target_valid=empty_series,
            random_state=0,
        )

    def info(self):
        """Display information about all data splits.

        Prints DataFrame info for main, test, train, and validation datasets.
        """
        self.features_main.info()
        self.features_test.info()
        self.features_train.info()
        self.features_valid.info()

    def with_upsampled_training(self) -> DataSplits:
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
        features_zeros = self.features_train[self.target_train == 0]
        features_ones = self.features_train[self.target_train == 1]
        target_zeros = self.target_train[self.target_train == 0]
        target_ones = self.target_train[self.target_train == 1]
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
            [features_maj] + [features_min] * increase_factor)
        target_upsampled = pd.concat(
            [target_maj] + [target_min] * increase_factor)

        # Shuffle and explicitly convert back to DataFrame/Series
        shuffled_features, shuffled_target = shuffle(  # type: ignore[misc]
            features_upsampled,
            target_upsampled,
            random_state=self.random_state
        )

        # Explicitly convert to proper types for dataclass constructor
        features_train_upsampled = pd.DataFrame(
            shuffled_features,  # type: ignore[arg-type]
            columns=features_upsampled.columns
        )

        target_train_upsampled = pd.Series(
            shuffled_target,  # type: ignore[arg-type]
            name=target_upsampled.name
        )

        return DataSplits(
            features_to_include=self.features_to_include,
            target_column=self.target_column,
            features_main=self.features_main,
            target_main=self.target_main,
            features_test=self.features_test,
            target_test=self.target_test,
            features_train=features_train_upsampled,
            target_train=target_train_upsampled,
            features_valid=self.features_valid,
            target_valid=self.target_valid,
            random_state=self.random_state
        )

    def with_downsampled_training(self) -> DataSplits:
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
        features_zeros = self.features_train[self.target_train == 0]
        features_ones = self.features_train[self.target_train == 1]
        target_zeros = self.target_train[self.target_train == 0]
        target_ones = self.target_train[self.target_train == 1]
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
            [features_min] + [features_maj.sample(frac=decrease_factor, random_state=self.random_state)])
        target_downsampled = pd.concat(
            [target_min] + [target_maj.sample(frac=decrease_factor, random_state=self.random_state)])

        shuffled_features, shuffled_target = shuffle(  # type: ignore[misc]
            features_downsampled,
            target_downsampled,
            random_state=self.random_state
        )

        # Explicitly convert to proper types for dataclass constructor
        features_train_downsampled = pd.DataFrame(
            shuffled_features,  # type: ignore[arg-type]
            columns=features_downsampled.columns
        )

        target_train_downsampled = pd.Series(
            shuffled_target,  # type: ignore[arg-type]
            name=target_downsampled.name
        )

        return DataSplits(
            features_to_include=self.features_to_include,
            target_column=self.target_column,
            features_main=self.features_main,
            target_main=self.target_main,
            features_test=self.features_test,
            target_test=self.target_test,
            features_train=features_train_downsampled,
            target_train=target_train_downsampled,
            features_valid=self.features_valid,
            target_valid=self.target_valid,
            random_state=self.random_state
        )

    def with_feature_subset(
            self,
            feature_subset: list[str]
    ) -> DataSplits:
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
            features_main=pd.DataFrame(self.features_main[feature_subset]),
            target_main=self.target_main,  # Reuse same Series
            features_test=pd.DataFrame(self.features_test[feature_subset]),
            target_test=self.target_test,  # Reuse same Series
            features_train=pd.DataFrame(self.features_train[feature_subset]),
            target_train=self.target_train,  # Reuse same Series
            features_valid=pd.DataFrame(self.features_valid[feature_subset]),
            target_valid=self.target_valid,  # Reuse same Series
            random_state=self.random_state
        )

    def auc_roc_curve(
            self,
            test_proba: np.ndarray,
            plot_title: str = 'ROC Curve'
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

        fpr, tpr, thresholds = roc_curve(self.target_test, test_proba)
        auc_score = float(auc(fpr, tpr))

        plt.figure(figsize=(8, 6))

        plt.plot(
            fpr,
            tpr,
            label=f'Model (AUC = {auc_score:.4f})',
            linewidth=2
        )

        # ROC curve for random model (looks like a straight line)
        plt.plot([0, 1], [0, 1], linestyle='--',
                 color='gray', label='Random Classifier')

        # Annotate threshold points
        indices = [0, len(thresholds)//4, len(thresholds) //
                   2, len(thresholds)-1]
        for idx, i in enumerate(indices):
            plt.scatter(fpr[i], tpr[i], color='red', s=50, zorder=5)
            plt.annotate(f'{thresholds[i]:.2f}',
                         xy=(fpr[i], tpr[i]),
                         xytext=(10, 10 + idx*15),
                         textcoords='offset points',
                         fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.title(plot_title)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.show()

        return auc_score
