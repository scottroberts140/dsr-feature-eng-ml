from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Union, Optional, cast, Mapping, Type, Any, Final
from dsr_feature_eng_ml.enums import ModelGeneralization, ModelType, ModelBalancing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError


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
            features: list[str],
            model: Union[DecisionTreeClassifier, RandomForestClassifier],
    ):
        try:
            self.feature_importances = pd.DataFrame(
                {
                    'feature': features,
                    'importance': model.feature_importances_
                }
            ).sort_values('importance', ascending=False)
        except (AttributeError, NotFittedError):
            self.feature_importances = pd.DataFrame(
                {
                    'feature': [],
                    'importance': np.empty(0, dtype=np.float64)
                }
            )

        self.features = self.feature_importances['feature'].to_list()
        self.feature_importances['cumulative_importance'] = self.feature_importances['importance'].cumsum(
        )
        self.threshold_80_idx = 0
        self.threshold_95_idx = 0

    @classmethod
    def empty(
        cls,
        model_type: ModelType
    ) -> ModelFeatureImportance:
        """Create an empty placeholder instance.

        Returns a sentinel instance with default values, typically used for
        initialization before actual feature importances are computed.

        Args:
            model_type (ModelType): The type of model (DecisionTree or RandomForest).

        Returns:
            ModelFeatureImportance: Empty instance with an empty feature list, and
            an uninitialized classifier.
        """
        features: list[str] = []

        match model_type:
            case ModelType.Decision_Tree:
                return cls(
                    features=features,
                    model=DecisionTreeClassifier()
                )
            case _:
                return cls(
                    features=features,
                    model=RandomForestClassifier()
                )

    def info(self) -> str:
        """Display formatted feature importance information.

        Prints each feature with its importance score and cumulative importance
        percentage.
        """
        retval: str = ''

        for i in range(len(self.feature_importances)):
            feature = self.feature_importances.iloc[i]['feature']
            importance = self.feature_importances.iloc[i]['importance']
            cumulative_importance = self.feature_importances.iloc[i]['cumulative_importance']
            retval += "{:<3} {:<20} Importance: {:.4f}   {:>8.2%}\n".format(
                i+1, feature, importance, cumulative_importance)

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
            cumulative_importance = self.feature_importances.iloc[n -
                                                                  1]['cumulative_importance']

            if self.threshold_80_idx == 0 and cumulative_importance >= 0.8:
                self.threshold_80_idx = n

            if self.threshold_95_idx == 0 and cumulative_importance >= 0.95:
                self.threshold_95_idx = n + 1
                break

        if self.threshold_95_idx > feature_count:
            self.threshold_95_idx = feature_count
