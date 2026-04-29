"""XGBoost classifier model specification and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dsr_utils import format_label_value_pairs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier as SklearnXGBClassifier

from dsr_feature_eng_ml.enums import (
    BalancingStrategy,
    ModelType,
    OptimizationStrategy,
    ScoringMetric,
    TaskType,
)
from dsr_feature_eng_ml.models.model_specification import (
    ClassificationModelParams,
    ClassificationModelSpecification,
)
from dsr_feature_eng_ml.prefs_instance import prefs


class EncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBClassifier adapter that supports non-numeric class labels.

    XGBoost's sklearn API expects class labels to be integer-encoded
    (0..K-1). This adapter encodes labels at fit-time and decodes predictions
    back to the original label space so the rest of the audit pipeline can
    continue to use native labels (e.g., "<=50K", ">50K").
    """

    def __init__(self, **xgb_params: Any):
        self.xgb_params = dict(xgb_params)
        self._model: SklearnXGBClassifier | None = None
        self._label_encoder: LabelEncoder | None = None
        self.classes_: Any = None

    def _get_fitted_model(self) -> SklearnXGBClassifier:
        if self._model is None:
            raise RuntimeError("Estimator has not been fitted yet.")
        return self._model

    @property
    def feature_importances_(self) -> Any:
        """Expose feature importances from the fitted XGBoost model."""
        return self._get_fitted_model().feature_importances_

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> EncodedXGBClassifier:
        """Fit XGBoost after label-encoding target classes."""
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        self._model = SklearnXGBClassifier(**self.xgb_params)
        if sample_weight is None:
            self._model.fit(X, y_encoded)
        else:
            self._model.fit(X, y_encoded, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> Any:
        """Predict class labels in the original label space."""
        model = self._get_fitted_model()
        enc = model.predict(X)
        if self._label_encoder is None:
            return enc
        return self._label_encoder.inverse_transform(enc.astype(int))

    def predict_proba(self, X: Any) -> Any:
        """Predict class probabilities."""
        return self._get_fitted_model().predict_proba(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters for sklearn clone/grid search."""
        return dict(self.xgb_params)

    def set_params(self, **params: Any) -> EncodedXGBClassifier:
        """Set estimator parameters for sklearn clone/grid search."""
        self.xgb_params.update(params)
        if self._model is not None:
            self._model.set_params(**params)
        return self


@dataclass(frozen=True)
class XGBClassifierParams(ClassificationModelParams):
    """
    Hyperparameters for XGBoost classifier models.

    XGBoost is a gradient-boosted decision tree ensemble method that is
    highly effective for tabular classification tasks, offering regularization,
    early stopping, and robust performance across many domains.
    """

    n_estimators: int | list[int] = 100
    max_depth: int | list[int] = 6
    learning_rate: float | list[float] = 0.3
    subsample: float | list[float] = 1.0
    colsample_bytree: float | list[float] = 1.0
    reg_alpha: float | list[float] = 0.0  # L1
    reg_lambda: float | list[float] = 1.0  # L2
    scale_pos_weight: float = 1.0
    use_label_encoder: bool = False
    eval_metric: str = "logloss"
    random_state: int | None = None

    def info(self) -> str:
        """Return a formatted summary of XGBoost parameters."""
        data = [
            ("Estimators", f"{self.n_estimators}"),
            ("Max Depth", f"{self.max_depth}"),
            ("Learning Rate", f"{self.learning_rate}"),
            ("Subsample", f"{self.subsample}"),
            ("Col Sample / Tree", f"{self.colsample_bytree}"),
            ("L1 (alpha)", f"{self.reg_alpha}"),
            ("L2 (lambda)", f"{self.reg_lambda}"),
            ("Scale Pos Weight", f"{self.scale_pos_weight}"),
        ]
        return format_label_value_pairs(data)

    @staticmethod
    def get_standard_search_grid(narrow: bool = True) -> dict[str, list[Any]]:
        """
        Generate a standard hyperparameter search grid for XGBoost.

        Parameters
        ----------
        narrow : bool, default True
            If True, returns a compact grid targeting common defaults.
            If False, returns an expanded grid.

        Returns
        -------
        dict[str, list[Any]]
            Parameter grid mapping keys to candidate values.
        """
        if narrow:
            return {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.1, 0.3],
            }

        return {
            "n_estimators": [50, 100, 200, 400],
            "max_depth": [3, 5, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        }


class XGBClassifierModel(
    ClassificationModelSpecification[XGBClassifierParams, EncodedXGBClassifier]
):
    """
    XGBoost classifier model specification.

    Manages the lifecycle of an XGBoost classifier, providing standardized
    fitting, tuning, and evaluation through the audit pipeline.
    """

    params_class = XGBClassifierParams

    def __init__(
        self,
        cv: int | None,
        balancing_strategy: BalancingStrategy = BalancingStrategy.NONE,
        params: XGBClassifierParams | None = None,
        task_type: TaskType = TaskType.CLASSIFICATION,
        scoring: ScoringMetric = ScoringMetric.F1,
        n_jobs: int = 3,
        n_iter: int = -1,
        acceptable_gap: float = prefs.acceptable_gap,
        large_gap: float = prefs.large_gap,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.MANUAL,
    ):
        """Initialize the XGBoost classifier model specification."""
        if params is None:
            params = XGBClassifierParams(
                task_type=task_type,
                random_state=1,
                scoring=scoring,
            )

        self._model_dials = params
        self._scoring = self.model_dials.scoring

        super().__init__(
            cv=cv,
            balancing_strategy=balancing_strategy,
            n_jobs=n_jobs,
            n_iter=n_iter,
            acceptable_gap=acceptable_gap,
            large_gap=large_gap,
            optimization_strategy=optimization_strategy,
        )

        self.estimator = self.create_estimator()

    @property
    def model_type(self) -> ModelType:
        """The XGBoost classifier model type identifier."""
        return ModelType.XGB_CLASSIFIER

    @property
    def scoring(self) -> ScoringMetric:
        """The scoring metric used for optimization."""
        return self._scoring

    @scoring.setter
    def scoring(self, value: ScoringMetric) -> None:
        self._scoring = value

    @property
    def model_dials(self) -> XGBClassifierParams:
        """The hyperparameter container for the current model."""
        return self._model_dials

    @model_dials.setter
    def model_dials(self, value: XGBClassifierParams) -> None:
        self._model_dials = value

    def get_estimator_class(self) -> type[EncodedXGBClassifier]:
        """Return the XGBoost adapter class with label encoding support."""
        return EncodedXGBClassifier

    def create_estimator(
        self, parameters: XGBClassifierParams | None = None
    ) -> EncodedXGBClassifier:
        """
        Instantiate a raw XGBClassifier estimator.

        Parameters
        ----------
        parameters : XGBClassifierParams, optional
            Parameter override. If None, uses instance dials.
        """
        p = parameters or self.model_dials

        n_estimators = (
            p.n_estimators[0] if isinstance(p.n_estimators, list) else p.n_estimators
        )
        max_depth = p.max_depth[0] if isinstance(p.max_depth, list) else p.max_depth
        learning_rate = (
            p.learning_rate[0] if isinstance(p.learning_rate, list) else p.learning_rate
        )
        subsample = p.subsample[0] if isinstance(p.subsample, list) else p.subsample
        colsample_bytree = (
            p.colsample_bytree[0]
            if isinstance(p.colsample_bytree, list)
            else p.colsample_bytree
        )
        reg_alpha = p.reg_alpha[0] if isinstance(p.reg_alpha, list) else p.reg_alpha
        reg_lambda = p.reg_lambda[0] if isinstance(p.reg_lambda, list) else p.reg_lambda

        return EncodedXGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=p.scale_pos_weight,
            eval_metric=p.eval_metric,
            random_state=p.random_state,
            n_jobs=self.n_jobs,
        )
