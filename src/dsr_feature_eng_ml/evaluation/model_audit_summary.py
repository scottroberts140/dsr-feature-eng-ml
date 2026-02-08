from __future__ import annotations
from dataclasses import dataclass, field
import dataclasses
from typing import Optional, TYPE_CHECKING, Any, List, Tuple, Set
import pandas as pd
import numpy as np
from dsr_feature_eng_ml.evaluation.schema import (
    ModelConfiguration,
    FeatureMetadata,
    DataSplits,
)
from dsr_feature_eng_ml.models import ModelSpecification
from dsr_feature_eng_ml.enums import ModelType
from dsr_feature_eng_ml.preferences import prefs
from dsr_utils.formatting import (
    FormatConfig,
    IntegerFormat,
    DateTimeFormat,
    format_label_value_pairs,
    format_as_grid,
)
from dsr_utils.enums import StringCase
from dsr_utils.strings import (
    convert_list_to_case,
    func_for_string_conv,
    to_original_string,
)
from dsr_files.excel_handler import save_excel, ExcelSheetConfig
from dsr_files.joblib_handler import save_joblib
from dsr_files.json_handler import save_json
from dsr_files.csv_handler import save_csv

import pandas as pd
from pathlib import Path
from datetime import datetime

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.table import Table
    from matplotlib.transforms import Bbox
    from dsr_files.enums import FileType

AUDIT_ANOMALY_ACTUAL_COL = "_audit_Actual"
AUDIT_ANOMALY_PREDICTED_COL = "_audit_Predicted"
AUDIT_ANOMALY_ABS_ERROR_COL = "_audit_Abs_Error"
AUDIT_ANOMALY_ACTUAL_COL_HEADER = "Actual"
AUDIT_ANOMALY_PREDICTED_COL_HEADER = "Predicted"
AUDIT_ANOMALY_ABS_ERROR_COL_HEADER = "Abs_Error"


class ModelAuditSummary:
    @dataclass
    class ModelPredictions:
        preds_val: Optional[pd.Series] = None
        probs_val: Optional[pd.DataFrame] = None
        preds_test: Optional[pd.Series] = None
        probs_test: Optional[pd.DataFrame] = None

    @property
    def solid_color_palette(self) -> dict:
        return self._solid_color_palette

    @property
    def light_color_palette(self) -> dict:
        return self._light_color_palette

    @property
    def random_state(self) -> int:
        return (
            self.data_splits.random_state
            if self.data_splits is not None
            else prefs.random_state
        )

    @property
    def data_splits(self) -> DataSplits:
        return self._data_splits

    @property
    def results(self) -> list[ModelConfiguration]:
        return self._results

    @results.setter
    def results(self, val: list[ModelConfiguration]) -> None:
        self._results = val

    @property
    def audit_timestamp(self) -> str:
        return self._audit_timestamp

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def original_row_count(self) -> int:
        return self._original_row_count

    @original_row_count.setter
    def original_row_count(self, val: int) -> None:
        self._original_row_count = val

    @property
    def cleaned_row_count(self) -> int:
        return self._cleaned_row_count

    @cleaned_row_count.setter
    def cleaned_row_count(self, val: int) -> None:
        self._cleaned_row_count = val

    @property
    def dropped_row_count(self) -> int:
        return self._dropped_row_count

    @dropped_row_count.setter
    def dropped_row_count(self, val: int) -> None:
        self._dropped_row_count = val

    @property
    def row_loss_pct(self) -> float:
        return self._row_loss_pct

    @row_loss_pct.setter
    def row_loss_pct(self, val: float) -> None:
        self._row_loss_pct = val

    @property
    def train_row_count(self) -> int:
        return self._train_row_count

    @train_row_count.setter
    def train_row_count(self, val: int) -> None:
        self._train_row_count = val

    @property
    def val_row_count(self) -> int:
        return self._val_row_count

    @val_row_count.setter
    def val_row_count(self, val: int) -> None:
        self._val_row_count = val

    @property
    def processed_row_count(self) -> int:
        return self._processed_row_count

    @processed_row_count.setter
    def processed_row_count(self, val: int) -> None:
        self._processed_row_count = val

    @property
    def test_row_count(self) -> int:
        return self._test_row_count

    @test_row_count.setter
    def test_row_count(self, val: int) -> None:
        self._test_row_count = val

    @property
    def top_n_importance(self) -> int:
        return self._top_n_importance

    @top_n_importance.setter
    def top_n_importance(self, val: int) -> None:
        self._top_n_importance = val

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, val: float) -> None:
        self._duration = val

    @property
    def features(self) -> dict[str, FeatureMetadata]:
        return self._features

    @property
    def features_used(self) -> List[FeatureMetadata]:
        return self._features_used

    @property
    def features_to_fit_set(self) -> Set[FeatureMetadata]:
        return self._features_to_fit_set

    @property
    def top_n_anomalies(self) -> int:
        return self._top_n_anomalies

    @top_n_anomalies.setter
    def top_n_anomalies(self, val: int) -> None:
        self._top_n_anomalies = val

    @property
    def anomaly_display_map(self) -> dict:
        return self._anomaly_display_map

    @property
    def actual_value_fmt(self) -> FormatConfig:
        return self._actual_value_fmt

    @actual_value_fmt.setter
    def actual_value_fmt(self, val: FormatConfig) -> None:
        self._actual_value_fmt = val

    @property
    def predicted_value_fmt(self) -> FormatConfig:
        return self._predicted_value_fmt

    @predicted_value_fmt.setter
    def predicted_value_fmt(self, val: FormatConfig) -> None:
        self._predicted_value_fmt = val

    @property
    def abs_error_fmt(self) -> FormatConfig:
        return self._abs_error_fmt

    @abs_error_fmt.setter
    def abs_error_fmt(self, val: FormatConfig) -> None:
        self._abs_error_fmt = val

    @property
    def error_pct_fmt(self) -> FormatConfig:
        return self._error_pct_fmt

    @error_pct_fmt.setter
    def error_pct_fmt(self, val: FormatConfig) -> None:
        self._error_pct_fmt = val

    @property
    def toc_registry(self) -> list:
        return self._toc_registry

    @property
    def anomaly_data(self) -> Optional[pd.DataFrame]:
        return self._anomaly_data

    @anomaly_data.setter
    def anomaly_data(self, val: Optional[pd.DataFrame]) -> None:
        self._anomaly_data = val

    @property
    def anomaly_dynamic_features(self) -> List[str]:
        return self._anomaly_dynamic_features

    @anomaly_dynamic_features.setter
    def anomaly_dynamic_features(self, val: List[str]) -> None:
        self._anomaly_dynamic_features = val

    @property
    def anomaly_threshold(self) -> float:
        return self._anomaly_threshold

    @anomaly_threshold.setter
    def anomaly_threshold(self, val: float) -> None:
        self._anomaly_threshold = val

    @property
    def anomaly_risk_concentration_threshold(self) -> float:
        return self._anomaly_risk_concentration_threshold

    @anomaly_risk_concentration_threshold.setter
    def anomaly_risk_concentration_threshold(self, val: float) -> None:
        self._anomaly_risk_concentration_threshold = val

    @property
    def model_accuracy_limit(self) -> float:
        return self._model_accuracy_limit

    @model_accuracy_limit.setter
    def model_accuracy_limit(self, val: float) -> None:
        self._model_accuracy_limit = val

    @property
    def model_acceptable_limit(self) -> float:
        return self._model_acceptable_limit

    @model_acceptable_limit.setter
    def model_acceptable_limit(self, val: float) -> None:
        self._model_acceptable_limit = val

    @property
    def model_stability_limit(self) -> float:
        return self._model_stability_limit

    @model_stability_limit.setter
    def model_stability_limit(self, val: float) -> None:
        self._model_stability_limit = val

    @property
    def model_efficiency_threshold(self) -> int:
        return self._model_efficiency_threshold

    @model_efficiency_threshold.setter
    def model_efficiency_threshold(self, val: int) -> None:
        self._model_efficiency_threshold = val

    @property
    def drift_threshold(self) -> float:
        return self._drift_threshold

    @drift_threshold.setter
    def drift_threshold(self, val: float) -> None:
        self._drift_threshold = val

    def __init__(
        self,
        data_splits: DataSplits,
        results: list[ModelConfiguration] = [],
        audit_timestamp: str = "",
        dataset_name: str = "Unknown Dataset",
        original_row_count: int = 0,
        cleaned_row_count: int = 0,
        dropped_row_count: int = 0,
        row_loss_pct: float = 0.0,
        train_row_count: int = 0,
        val_row_count: int = 0,
        processed_row_count: int = 0,
        test_row_count: int = 0,
        top_n_importance: int = 1,
        duration: float = 0.0,
        features: dict[str, FeatureMetadata] = {},
        top_n_anomalies: int = 5,
        anomaly_display_map: dict = {},
        actual_value_fmt: Any = None,
        predicted_value_fmt: Any = None,
        abs_error_fmt: Any = None,
        error_pct_fmt: Any = None,
        toc_registry=[],
        anomaly_data: Optional[pd.DataFrame] = None,
        anomaly_dynamic_features: List[str] = [],
        anomaly_threshold: float = prefs.anomaly_threshold,
        anomaly_risk_concentration_threshold: float = (
            prefs.anomaly_risk_concentration_threshold
        ),
        model_accuracy_limit: float = prefs.model_accuracy_limit,
        model_acceptable_limit: float = prefs.model_acceptable_limit,
        model_stability_limit: float = prefs.model_stability_limit,
        model_efficiency_threshold: int = prefs.model_efficiency_threshold,
        drift_threshold: float = prefs.drift_threshold,
    ) -> None:
        self._data_splits = data_splits
        self._results = results
        self._audit_timestamp = audit_timestamp
        self._dataset_name = dataset_name
        self._original_row_count = original_row_count
        self._cleaned_row_count = cleaned_row_count
        self._dropped_row_count = dropped_row_count
        self._row_loss_pct = row_loss_pct
        self._train_row_count = train_row_count
        self._val_row_count = val_row_count
        self._processed_row_count = processed_row_count
        self._test_row_count = test_row_count
        self._top_n_importance = top_n_importance
        self._duration = duration
        self._features = features
        self._top_n_anomalies = top_n_anomalies
        self._anomaly_display_map = anomaly_display_map
        self._actual_value_fmt = actual_value_fmt
        self._predicted_value_fmt = predicted_value_fmt
        self._abs_error_fmt = abs_error_fmt
        self._error_pct_fmt = error_pct_fmt
        self._toc_registry = toc_registry
        self._anomaly_data = anomaly_data
        self._anomaly_dynamic_features = anomaly_dynamic_features
        self._anomaly_threshold = anomaly_threshold
        self._anomaly_risk_concentration_threshold = (
            anomaly_risk_concentration_threshold
        )
        self._model_accuracy_limit = model_accuracy_limit
        self._model_acceptable_limit = model_acceptable_limit
        self._model_stability_limit = model_stability_limit
        self._model_efficiency_threshold = model_efficiency_threshold
        self._drift_threshold = drift_threshold

        # Create timestamp (Format: 20251227_1935) when not provided
        if not self._audit_timestamp:
            timestamp_format = DateTimeFormat(
                date_format="%Y%m%d", time_format="%H%M", separator="_"
            )
            self._audit_timestamp = timestamp_format.format_value(datetime.now())
        self._features_used = list(self.features.values())
        self._features_used = sorted(
            self._features_used, key=lambda fm: fm.name.lower()
        )
        self._solid_color_palette: dict = prefs.get_solid_palette()
        self._light_color_palette: dict = prefs.get_light_palette()
        self._init_features_to_fit_set()

    def __setstate__(self, state):
        # This logic runs when the object is being 'unpickled'
        # Update the internal dictionary with the loaded state
        self.__dict__.update(state)

        new_results: list[ModelConfiguration] = []
        for config in self.results:
            print(f"Preprocessing model: {config.model_type.value}")
            model_spec = ModelSpecification.create_model_from_config(config)
            val_result: Optional[ModelConfiguration] = None
            test_result: Optional[ModelConfiguration] = None
            new_result: Optional[ModelConfiguration] = None
            if config.has_val_set_evaluation_scores and config.preds_val is None:
                if model_spec is not None:
                    val_result = model_spec.fit_and_evaluate_val(
                        data_splits=self.data_splits,
                        id=config.id,
                        features_to_fit_set=self.features_to_fit_set,
                        score_cv=config.score_cv,
                        use_combined_data=config.use_combined_data,
                        filter_outliers=config.filter_outliers,
                        outlier_count=config.outlier_count,
                    )
                    if val_result is not None:
                        new_result = dataclasses.replace(
                            config,
                            score_train=val_result.score_train,
                            score_val=val_result.score_val,
                            score_val_cleaned=val_result.score_val_cleaned,
                            mae_train=val_result.mae_train,
                            mae_val=val_result.mae_val,
                            mse_train=val_result.mse_train,
                            mse_val=val_result.mse_val,
                            r2_train=val_result.r2_train,
                            r2_val=val_result.r2_val,
                            r2_val_cleaned=val_result.r2_val_cleaned,
                            accuracy_train=val_result.accuracy_train,
                            accuracy_val=val_result.accuracy_val,
                            accuracy_val_cleaned=val_result.accuracy_val_cleaned,
                            preds_val=val_result.preds_val,
                            probs_val=val_result.probs_val,
                        )

                if config.has_test_set_evaluation_scores and config.preds_test is None:
                    if model_spec is not None:
                        test_result = model_spec.evaluate_test_set_performance(
                            data_splits=self.data_splits,
                            config=config,
                            features_to_fit_set=self.features_to_fit_set,
                        )
                        if test_result is not None:
                            if new_result is not None:
                                new_result = dataclasses.replace(
                                    new_result,
                                    has_test_set_evaluation_scores=True,
                                    score_test=test_result.score_test,
                                    mae_test=test_result.mae_test,
                                    mse_test=test_result.mse_test,
                                    r2_test=test_result.r2_test,
                                    accuracy_test=test_result.accuracy_test,
                                    preds_test=test_result.preds_test,
                                    probs_test=test_result.probs_test,
                                    test_mean=test_result.test_mean,
                                    test_std=test_result.test_std,
                                    test_median=test_result.test_median,
                                    test_skew=test_result.test_skew,
                                    test_kurtosis=test_result.test_kurtosis,
                                )
                            else:
                                new_result = test_result

            if new_result is not None:
                new_results.append(new_result)
            else:
                new_results.append(config)

        self.results = new_results

        # Check for attributes that might be missing from older versions
        # if "anomaly_threshold" not in state:
        #    self.anomaly_threshold = prefs.anomaly_threshold

    def _init_features_to_fit_set(self) -> None:
        self._features_to_fit_set = FeatureMetadata.dict_to_set(
            feature_dict=self.features,
            target_column=self.data_splits.target_column,
        )

    @classmethod
    def from_joblib(cls, filepath: Path) -> "ModelAuditSummary":
        from dsr_files.joblib_handler import load_joblib

        loaded_data = load_joblib(filepath=filepath)

        # If it's already an instance
        if not isinstance(loaded_data, cls):
            raise TypeError(f"File at {filepath} is a {type(loaded_data)}, not {cls}")

        return loaded_data

    def add_model_configuration(self, config: ModelConfiguration) -> None:
        """Adds a completed audit snapshot."""
        self.results.append(config)

    @property
    def best_overall_model(self) -> Optional[ModelConfiguration]:
        """Returns the highest scoring model regardless of type."""
        if not self.results:
            return None
        # Uses the __lt__ implementation in ModelConfiguration for comparison
        return max(self.results)

    def _calculate_row_counts(self):
        ds = self.data_splits
        self.train_row_count = len(ds.train_features)
        self.val_row_count = len(ds.val_features)
        self.processed_row_count = self.train_row_count + self.val_row_count
        self.test_row_count = len(ds.test_features)
        self.cleaned_row_count = (
            self.train_row_count + self.val_row_count + self.test_row_count
        )
        self.dropped_row_count = ds.original_row_count - self.cleaned_row_count
        self.row_loss_pct = (self.dropped_row_count / ds.original_row_count) * 100

    def get_state(self, include_preds: bool = False) -> dict:
        """Exports the entire audit snapshot for persistence."""
        import numpy as np

        # Calculate aggregate stats
        total_cpu_time = sum(res.total_duration for res in self.results)
        max_ram_observed = max(res.actual_peak_gb for res in self.results)
        total_fits = sum(res.num_candidates * res.cv for res in self.results)

        # Calculate avg efficiency safely
        efficiencies = [
            res.efficiency(data_splits=self.data_splits) for res in self.results
        ]
        efficiencies = [e for e in efficiencies if e > 0]
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0

        return {
            "metadata": {
                "timestamp": self.audit_timestamp,
                "row_counts": {
                    "train": self.train_row_count,
                    "val": self.val_row_count,
                    "test": self.test_row_count,
                    "original": self.original_row_count,
                    "cleaned": self.cleaned_row_count,
                    "dropped": self.dropped_row_count,
                    "row_loss_pct": self.row_loss_pct,
                    "processed": self.processed_row_count,
                },
                "top_n_importance": self.top_n_importance,
                "dataset_name": self.dataset_name,
                "duration": self.duration,
                "anomalies": {
                    "threshold": self.anomaly_threshold,
                    "display_map": self.anomaly_display_map,
                    "data": (
                        self.anomaly_data
                        if self.anomaly_data is not None
                        else pd.DataFrame()
                    ),
                    "dynamic_features": self.anomaly_dynamic_features,
                    "top_n": self.top_n_anomalies,
                    "formats": {
                        "actual": self.actual_value_fmt.to_dict(),
                        "predicted": self.predicted_value_fmt.to_dict(),
                        "abs_error": self.abs_error_fmt.to_dict(),
                    },
                },
                "model_limits": {
                    "accuracy": self.model_accuracy_limit,
                    "acceptable": self.model_acceptable_limit,
                    "stability": self.model_stability_limit,
                    "efficiency_threshold": self.model_efficiency_threshold,
                },
                "aggregate_stats": {
                    "total_computational_time": total_cpu_time,
                    "peak_system_ram_usage_gb": max_ram_observed,
                    "total_cv_fits_performed": total_fits,
                    "average_efficiency_rows_per_sec": avg_efficiency,
                },
                "features": {
                    key: feature.to_dict() for key, feature in self.features.items()
                },
            },
            "results": [
                config.to_dict(include_preds=include_preds) for config in self.results
            ],
        }

    def capture_anomaly_context(self) -> None:
        """
        Identifies and stores the top outliers and high-kurtosis features
        for the best performing model.
        """
        config = self.best_overall_model

        if config is None:
            return

        X_val = self.data_splits.val_features
        y_val = self.data_splits.val_target
        preds_val = config.preds_val if config.preds_val is not None else pd.Series()
        # Re-calculate or retrieve the top n anomalies absolute errors
        abs_errors = np.abs(y_val.to_numpy().flatten() - preds_val.to_numpy().flatten())
        threshold = np.percentile(abs_errors, self.anomaly_threshold)
        anomaly_mask = abs_errors >= threshold

        anomalies_scaled = X_val.iloc[anomaly_mask].copy()
        anomalies = self.data_splits.inverse_transform_df(anomalies_scaled)
        anomalies[AUDIT_ANOMALY_ACTUAL_COL] = y_val.iloc[anomaly_mask].values
        anomalies[AUDIT_ANOMALY_PREDICTED_COL] = preds_val.iloc[anomaly_mask].values
        anomalies[AUDIT_ANOMALY_ABS_ERROR_COL] = abs_errors[anomaly_mask]

        # FILTER: Only look at numeric columns for kurtosis
        # This prevents the "Categorical does not support reduction kurt" error
        numeric_df = X_val.select_dtypes(include=[np.number])

        # Calculate kurtosis only on the numeric subset
        # Rank them to see which features have the most extreme outliers
        dynamic_features = (
            numeric_df.kurt()
            .sort_values(ascending=False)
            .index[: self.top_n_importance]
            .tolist()
        )

        self.anomaly_data = anomalies.sort_values(
            AUDIT_ANOMALY_ABS_ERROR_COL, ascending=False
        )
        self.anomaly_dynamic_features = dynamic_features

    def create_metadata(self) -> None:
        if len(self.results) == 0:
            return
        self._calculate_row_counts()

    def get_summary_data(
        self,
        config: ModelConfiguration,
        key_value_case: StringCase = StringCase.ORIGINAL,
    ) -> dict:
        import dataclasses

        if config.score_train is not None:
            train_score = config.score_train
        else:
            train_score = 0.0

        if config.score_val is not None:
            val_score = config.score_val
        else:
            val_score = 0.0

        if config.score_val_cleaned is not None:
            cleaned_score = config.score_val_cleaned
        else:
            cleaned_score = 0.0

        keys = [
            "ID",
            "Model",
            "Strategy",
            "Available RAM",
            "Est Peak RAM",
            "Actual Peak RAM",
            "Memory Risk",
            "Sampling %",
            "n_jobs",
            "Val Score",
            "Cleaned Score",
            "Total Duration",
            "Efficiency",
            "Train Score",
            "Gap",
            "Status",
            "Train Mean",
            "Train Std Dev",
            "Train Median",
            "Train Skew",
            "Train Kurtosis",
            "Val Mean",
            "Val Std Dev",
            "Val Median",
            "Val Skew",
            "Val Kurtosis",
            "Mean Delta",
            "Std Dev Delta",
            "Quality Score",
            "Drift Index",
        ]

        if key_value_case == StringCase.ORIGINAL:
            str_convert_func = to_original_string
        else:
            keys = convert_list_to_case(keys, key_value_case)
            str_convert_func = func_for_string_conv(key_value_case)

        # These values must be in the same order as the items in keys
        values = [
            config.id,
            str_convert_func(config.model_type.value),
            str_convert_func(config.balancing_strategy.value),
            config.available_gb,
            config.estimated_peak_gb,
            config.actual_peak_gb,
            bool(config.memory_risk_triggered),
            config.sampling_factor,
            config.concurrent_workers,
            val_score,
            cleaned_score,
            config.total_duration,
            config.efficiency,
            train_score,
            config.gap,
            str_convert_func(config.model_generalization.value),
            config.train_mean,
            config.train_std,
            config.train_median,
            config.train_skew,
            config.train_kurtosis,
            config.val_mean,
            config.val_std,
            config.val_median,
            config.val_skew,
            config.val_kurtosis,
            config.mean_delta,
            config.std_delta,
            config.quality_score,
            config.drift_index,
        ]
        data = {key: value for key, value in zip(keys, values)}
        return data

    def get_leaderboard(self) -> str:
        """Generates the summary table for the audit report using the centralized state."""
        # 1. Pull the already-calculated state
        state = self.get_state(include_preds=False)
        meta = state["metadata"]
        stats = meta["aggregate_stats"]

        # 2. Build the Console Header
        # We reconstruct the display pairs from the state dictionary
        row_count_format = IntegerFormat()
        row_counts = meta["row_counts"]
        display_pairs = [
            ("Timestamp", meta["timestamp"]),
            (
                "Train/Val/Test Rows",
                f"{row_count_format.format_value(row_counts['train'])} / {row_count_format.format_value(row_counts['val'])} / {row_count_format.format_value(row_counts['test'])}",
            ),
            ("Total Duration", stats["total_computational_time"]),
            ("Peak RAM", stats["peak_system_ram_usage_gb"]),
        ]

        # Format the features as a grid for the console
        grid_padding = max(len(f.name) for f in self.features_used) + 4
        features_grid = format_as_grid(
            input=[s.name for s in self.features_used],
            cols=3,
            padding=grid_padding,
            indent=4,
        )

        meta_header = (
            f"\n--- Audit Snapshot: {self.audit_timestamp} ---\n"
            f"{format_label_value_pairs(display_pairs, padding=4)}\n"
            f"    Features:\n{features_grid}\n"
            f"{'-' * prefs.report_width}"
        )

        # 3. Build the Table using existing row logic
        # We use results directly from the object to get the full list
        summary = [self.get_summary_data(config=config) for config in self.results]

        df = pd.DataFrame(summary).sort_values("Val Score", ascending=False)

        # Optional: Select a subset of columns for the console to prevent wrapping
        console_cols = [
            "ID",
            "Model",
            "Val Score",
            "Cleaned Score",
            "Gap",
            "Status",
            "Efficiency",
        ]

        return meta_header + "\n" + df[console_cols].to_string(index=False)

    def get_best_by_type(self, model_type: ModelType) -> Optional[ModelConfiguration]:
        type_results = [r for r in self.results if r.model_type == model_type]
        return max(type_results) if type_results else None

    # def get_transient_predictions(
    #     self, config: ModelConfiguration
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Re-generates predictions and probabilities on the fly without
    #     modifying the frozen config object.
    #     """
    #     model_spec = ModelSpecification.create_model_from_config(config)
    #     # Re-fit the model using the stored parameters and data_splits
    #     # Note: This is fast because we aren't tuning, just a single fit.
    #     if model_spec is not None:
    #         model_spec.fit(
    #             data_splits=self.data_splits,
    #             features_to_fit_set=self.features_to_fit_set,
    #         )

    #         preds = model_spec.predict(self.data_splits.val_features)
    #         probs = model_spec.predict_proba(self.data_splits.val_features)

    #     return preds, probs

    def _extract_preds_and_probs(self) -> List[ModelPredictions]:
        new_results: List[ModelConfiguration] = []
        model_predictions: List[ModelAuditSummary.ModelPredictions] = []

        for config in self.results:
            model_predictions.append(
                ModelAuditSummary.ModelPredictions(
                    preds_val=config.preds_val,
                    probs_val=config.probs_val,
                    preds_test=config.preds_test,
                    probs_test=config.probs_test,
                )
            )
            new_results.append(
                dataclasses.replace(
                    config,
                    preds_val=None,
                    probs_val=None,
                    preds_test=None,
                    probs_test=None,
                )
            )

        self.results = new_results
        return model_predictions

    def _restore_preds_and_probs(
        self, model_predictions: List[ModelPredictions]
    ) -> None:
        new_results: List[ModelConfiguration] = []
        for index in range(len(model_predictions)):
            model_prediction = model_predictions[index]
            config = self.results[index]
            new_results.append(
                dataclasses.replace(
                    config,
                    preds_val=model_prediction.preds_val,
                    probs_val=model_prediction.probs_val,
                    preds_test=model_prediction.preds_test,
                    probs_test=model_prediction.probs_test,
                )
            )

        self.results = new_results

    def export_results(
        self,
        prefix: str,
        file_type: FileType,
        path: Path,
        path_is_full_path: bool = False,
        append_timestamp_to_save_path: bool = False,
        report_title: str = "Model Audit Report",
    ) -> Path:
        from dsr_files.enums import FileType
        from dsr_files.json_handler import to_JSON_safe

        export_payload = {}
        export_payload["audit_id"] = f"{prefix}_{self.audit_timestamp}"
        export_payload["data_quality_score"] = self.results[0].quality_score

        if not path_is_full_path:
            output_path = Path(path)

            if append_timestamp_to_save_path:
                output_path = output_path / self.audit_timestamp

            filename = f"{prefix}_{self.audit_timestamp}"
        else:
            output_path = path.parent
            filename = path.stem

        full_path = Path()
        export_payload = self.get_state(include_preds=False)

        # Metadata file
        # Flatten the nested dicts like 'row_counts' and 'aggregate_stats'
        # so they look good in a two-column CSV.
        metadata = export_payload["metadata"]
        anomalies = metadata["anomalies"]
        anomaly_display_map = anomalies["display_map"]
        features = to_JSON_safe(metadata["features"])
        meta_extract = {
            "audit_id": f"{prefix}_{self.audit_timestamp}",
            "timestamp": metadata["timestamp"],
            **metadata["row_counts"],
            **metadata["aggregate_stats"],
            "features": features,
        }

        if FileType.CSV in file_type:
            df_metadata = pd.DataFrame(list(meta_extract.items()))
            _ = save_csv(
                data=df_metadata,
                filepath=output_path,
                filename=f"{filename}_metadata",
                header=False,
            )

            # Anomaly mapping
            df_anomaly_display_map = pd.DataFrame([anomaly_display_map])
            _ = save_csv(
                data=df_anomaly_display_map,
                filepath=output_path,
                filename=f"{filename}_anomaly_display_map",
                header=False,
            )

            # Anomaly data file
            df_anomaly_data = anomalies["data"]
            _ = save_csv(
                data=df_anomaly_data,
                filepath=output_path,
                filename=f"{filename}_anomaly_data",
            )

            # Anomaly dynanmic features file
            df_anomaly_dynamic_features = pd.DataFrame([anomalies["dynamic_features"]])
            _ = save_csv(
                data=df_anomaly_dynamic_features,
                filepath=output_path,
                filename=f"{filename}_anomaly_dynamic_features",
                header=False,
            )

            # Results file
            full_path = save_csv(
                data=pd.DataFrame(export_payload["results"]),
                filepath=output_path,
                filename=filename,
            )

        if FileType.JSON in file_type:
            full_path = save_json(
                data=export_payload, filepath=output_path, filename=filename
            )

        if FileType.JOBLIB in file_type:
            model_predictions: List[ModelAuditSummary.ModelPredictions] = (
                self._extract_preds_and_probs()
            )
            full_path = save_joblib(data=self, filepath=output_path, filename=filename)
            self._restore_preds_and_probs(model_predictions=model_predictions)

        if FileType.EXCEL in file_type:

            # Define the "Table of Contents" for the Excel file
            sheets = [
                # 1. High-level Metadata
                ExcelSheetConfig(
                    data=pd.DataFrame(list(meta_extract.items())),
                    sheet_name="Audit Summary",
                    header=False,
                ),
                # 2. Performance Leaderboard
                ExcelSheetConfig(
                    data=pd.DataFrame(export_payload["results"]),
                    sheet_name="Leaderboard",
                ),
                # 3. Anomalies (The "Smoking Gun")
                ExcelSheetConfig(data=anomalies["data"], sheet_name="Anomaly Log"),
                # 4. Technical Configuration
                ExcelSheetConfig(
                    data=pd.DataFrame(features),
                    sheet_name="Features Used",
                    index=True,
                ),
            ]

            save_excel(data=sheets, filepath=output_path, filename=filename)

        if FileType.PDF in file_type:
            from dsr_feature_eng_ml.evaluation.audit_pdf_renderer import (
                AuditPDFRenderer,
            )

            renderer = AuditPDFRenderer(summary=self, report_title=report_title)
            pdf_doc = renderer.render()
            pdf_doc.save(filepath=output_path, filename=filename)

        return full_path

    def evaluate_test_model(self, index: int, joblib_fullpath: Optional[Path]) -> None:
        config = self.results[index]
        print(f"Evaluating Test model {index}: {config.model_type.value}")
        model = ModelSpecification.create_model_from_config(config)

        if model is not None:
            self.results[index] = model.evaluate_test_set_performance(
                data_splits=self.data_splits,
                config=config,
                features_to_fit_set=self.features_to_fit_set,
            )

            if joblib_fullpath is not None:
                from dsr_files.enums import FileType

                _ = self.export_results(
                    prefix="Audit_State",
                    file_type=FileType.JOBLIB,
                    path=joblib_fullpath,
                    append_timestamp_to_save_path=False,
                    path_is_full_path=True,
                )
                print(f"JOBLIB file updated: {joblib_fullpath}")
        else:
            print(f"Unable to instantiate model for {config.model_type.name}")

    def evaluate_test_models(
        self, indexes: List[int], joblib_fullpath: Optional[Path]
    ) -> None:
        for index in indexes:
            self.evaluate_test_model(index=index, joblib_fullpath=joblib_fullpath)

    def evaluate_all_test_models(self, joblib_fullpath: Optional[Path]) -> None:
        self.evaluate_test_models(
            indexes=list(range(len(self.results))), joblib_fullpath=joblib_fullpath
        )
