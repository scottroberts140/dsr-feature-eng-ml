from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ValidationResults:
    """Store validation model predictions and metrics.

    Lightweight container for validation results without storing the full model.
    Stores predictions that can be reused for confusion matrix calculations
    and other analyses without needing to keep the trained model in memory.

    Attributes:
        predicted_valid (pd.Series): Predictions on validation set.
        predicted_train (pd.Series): Predictions on training set.
    """
    predicted_valid: pd.Series
    predicted_train: pd.Series

    @classmethod
    def empty(cls) -> ValidationResults:
        """Create an empty placeholder instance."""
        return cls(
            predicted_valid=pd.Series(dtype=int),
            predicted_train=pd.Series(dtype=int)
        )
