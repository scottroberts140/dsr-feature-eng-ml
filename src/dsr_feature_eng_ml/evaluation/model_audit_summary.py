"""Audit summary model and export utilities."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from dsr_files.csv_handler import save_csv
from dsr_files.excel_handler import ExcelSheetConfig, save_excel
from dsr_files.joblib_handler import save_joblib
from dsr_files.json_handler import save_json
from dsr_files.utils import PathLike
from dsr_utils.enums import StringCase
from dsr_utils.formatting import (
    DataFormat,
    DataScale,
    DateTimeFormat,
    FloatFormat,
    FormatConfig,
    IntegerFormat,
    StringFormat,
    format_as_grid,
    format_label_value_pairs,
)
from dsr_utils.strings import (
    convert_list_to_case,
    func_for_string_conv,
    to_original_string,
)

from dsr_feature_eng_ml.enums import ModelType, TaskType
from dsr_feature_eng_ml.models import ModelSpecification
from dsr_feature_eng_ml.prefs_instance import prefs

from .audit_pdf_renderer import AuditPDFRenderer
from .schema import DataSplits, FeatureMetadata, ModelConfiguration

if TYPE_CHECKING:
    from cloudpathlib import CloudPath
    from dsr_files.enums import FileType

# Constants for Anomaly Reporting
AUDIT_ANOMALY_ACTUAL_COL = "_audit_Actual"
AUDIT_ANOMALY_PREDICTED_COL = "_audit_Predicted"
AUDIT_ANOMALY_ABS_ERROR_COL = "_audit_Abs_Error"
AUDIT_ANOMALY_ACTUAL_COL_HEADER = "Actual"
AUDIT_ANOMALY_PREDICTED_COL_HEADER = "Predicted"
AUDIT_ANOMALY_ABS_ERROR_COL_HEADER = "Abs_Error"


class ModelAuditSummary:
    """
    Aggregate audit results, metrics, and export helpers.

    Acts as the primary data model for the final audit report, capturing
    performance across all models, anomaly context, and data distribution stats.
    """

    @dataclass
    class ModelPredictions:
        """Container for validation/test predictions and probabilities."""

        preds_val: pd.Series | None = None
        probs_val: pd.DataFrame | None = None
        preds_test: pd.Series | None = None
        probs_test: pd.DataFrame | None = None

    # --- Internal Attribute Declarations ---
    _features_to_fit_set: set[FeatureMetadata]
    _results: list[ModelConfiguration]
    _row_loss_pct: float = 0.0

    # --- Properties ---
    # Using public attributes for these reduces boilerplate while maintaining V1.2.0 style

    @property
    def random_state(self) -> int | None:
        """Return random state from splits or global preferences."""
        return self.data_splits.random_state if self.data_splits else prefs.random_state

    @property
    def row_loss_pct(self) -> float:
        """Percentage of rows lost during preprocessing."""
        return self._row_loss_pct

    @row_loss_pct.setter
    def row_loss_pct(self, val: float) -> None:
        """Allow manual or computed updates to row loss percentage."""
        self._row_loss_pct = float(val)

    def __init__(
        self,
        data_splits: DataSplits,
        results: list[ModelConfiguration] | None = None,
        audit_timestamp: str = "",
        dataset_name: str = "Unknown Dataset",
        original_row_count: int = 0,
        cleaned_row_count: int = 0,
        dropped_row_count: int = 0,
        train_row_count: int = 0,
        val_row_count: int = 0,
        test_row_count: int = 0,
        top_n_importance: int = 1,
        duration: float = 0.0,
        features: dict[str, FeatureMetadata] | None = None,
        top_n_anomalies: int = 5,
        anomaly_display_map: dict[str, str] | None = None,
        actual_value_fmt: FormatConfig | None = None,
        predicted_value_fmt: FormatConfig | None = None,
        abs_error_fmt: FormatConfig | None = None,
        error_pct_fmt: FormatConfig | None = None,
        anomaly_data: pd.DataFrame | None = None,
        anomaly_dynamic_features: list[str] | None = None,
        anomaly_threshold: float = prefs.anomaly_threshold,
        anomaly_risk_concentration_threshold: float = prefs.anomaly_risk_concentration_threshold,
        model_accuracy_limit: float = prefs.model_accuracy_limit,
        model_acceptable_limit: float = prefs.model_acceptable_limit,
        model_stability_limit: float = prefs.model_stability_limit,
        model_efficiency_threshold: int = prefs.model_efficiency_threshold,
        drift_threshold: float = prefs.drift_threshold,
    ) -> None:
        """Initialize the audit summary container."""
        self.data_splits = data_splits
        self.results = results or []
        self.dataset_name = dataset_name
        self.original_row_count = original_row_count
        self.cleaned_row_count = cleaned_row_count
        self.dropped_row_count = dropped_row_count
        self.train_row_count = train_row_count
        self.val_row_count = val_row_count
        self.test_row_count = test_row_count
        self.processed_row_count = train_row_count + val_row_count
        self.top_n_importance = top_n_importance
        self.duration = duration
        self.features = features or {}
        self.top_n_anomalies = top_n_anomalies
        self.anomaly_display_map = anomaly_display_map or {}
        self.actual_value_fmt = actual_value_fmt or StringFormat()
        self.predicted_value_fmt = predicted_value_fmt or StringFormat()
        self.abs_error_fmt = abs_error_fmt or FloatFormat()
        self.error_pct_fmt = error_pct_fmt or FloatFormat()
        self.anomaly_data = anomaly_data
        self.anomaly_dynamic_features = anomaly_dynamic_features or []
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_risk_concentration_threshold = anomaly_risk_concentration_threshold
        self.model_accuracy_limit = model_accuracy_limit
        self.model_acceptable_limit = model_acceptable_limit
        self.model_stability_limit = model_stability_limit
        self.model_efficiency_threshold = model_efficiency_threshold
        self.drift_threshold = drift_threshold

        # Initialize Audit Timestamp
        if not audit_timestamp:
            ts_fmt = DateTimeFormat(
                date_format="%Y%m%d", time_format="%H%M%S", separator="_"
            )
            self.audit_timestamp = ts_fmt.format_value(datetime.now())
        else:
            self.audit_timestamp = audit_timestamp

        # Sort features for consistent reporting
        self.features_used = sorted(
            list(self.features.values()), key=lambda fm: fm.name.lower()
        )

        self.solid_color_palette = prefs.get_solid_palette()
        self.light_color_palette = prefs.get_light_palette()

        # Internal state initialization
        self._init_features_to_fit_set()

    def _init_features_to_fit_set(self) -> None:
        """Initialize the features-to-fit set excluding the target column."""
        self._features_to_fit_set = FeatureMetadata.dict_to_set(
            feature_dict=self.features,
            target_column=self.data_splits.target_column,
        )

    def resolve_feature(self, col_name: str) -> FeatureMetadata | None:
        """Look up FeatureMetadata by name, falling back to parent for OHE columns.

        When one-hot encoding produces columns like ``DOLocationID_71``, this
        method finds the parent ``DOLocationID`` entry so formatters and short
        names are available even for encoded variants.
        """
        if col_name in self.features:
            return self.features[col_name]
        for key in self.features:
            if col_name.startswith(f"{key}_"):
                return self.features[key]
        return None

    @property
    def features_to_fit_set(self) -> set[FeatureMetadata]:
        """Set of features eligible for model fitting."""
        return self._features_to_fit_set

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore state and recompute missing prediction artifacts.

        On unpickle, if prediction or probability artifacts are missing from the
        config (common in older saves), they are recomputed to ensure the
        audit remains fully interactive.
        """
        self.__dict__.update(state)

        # Ensure feature set is initialized before re-running fits
        self._init_features_to_fit_set()

        new_results: list[ModelConfiguration] = []

        for config in self.results:
            model_spec = ModelSpecification.create_model_from_config(config)
            if model_spec is None:
                new_results.append(config)
                continue

            current_cfg = config

            # 1. Recompute Validation Artifacts if missing
            if config.has_val_set_evaluation_scores and config.preds_val is None:
                val_res = model_spec.fit_and_evaluate_val(
                    data_splits=self.data_splits,
                    id=config.id,
                    features_to_fit_set=self.features_to_fit_set,
                    score_cv=config.score_cv,
                    use_combined_data=config.use_combined_data,
                    filter_outliers=config.filter_outliers,
                    outlier_count=config.outlier_count,
                )
                if val_res:
                    current_cfg = dataclasses.replace(
                        current_cfg,
                        # Preserve persisted scalar metrics; only backfill
                        # missing prediction artifacts needed for reporting.
                        preds_val=val_res.preds_val,
                        probs_val=val_res.probs_val,
                    )

            # 2. Recompute Test Artifacts if missing
            if config.has_test_set_evaluation_scores and config.preds_test is None:
                test_res = model_spec.evaluate_test_set_performance(
                    data_splits=self.data_splits,
                    config=current_cfg,
                    features_to_fit_set=self.features_to_fit_set,
                )
                if test_res:
                    current_cfg = dataclasses.replace(
                        current_cfg,
                        # Preserve persisted scalar metrics; only backfill
                        # missing prediction artifacts needed for reporting.
                        preds_test=test_res.preds_test,
                        probs_test=test_res.probs_test,
                    )

            new_results.append(current_cfg)

        self.results = new_results

    @classmethod
    def from_joblib(cls, filepath: Path) -> ModelAuditSummary:
        """
        Load a saved audit summary from a joblib file.

        Parameters
        ----------
        filepath : Path
            The location of the .joblib file.

        Returns
        -------
        ModelAuditSummary
            The hydrated summary instance.
        """
        from dsr_files.joblib_handler import load_joblib

        loaded_data, _ = load_joblib(filepath=filepath)

        if not isinstance(loaded_data, cls):
            raise TypeError(
                f"File at {filepath} contains {type(loaded_data)}, "
                f"expected {cls.__name__}"
            )

        return loaded_data

    def add_model_configuration(self, config: ModelConfiguration) -> None:
        """Append a completed model configuration to the results list."""
        self.results.append(config)

    @property
    def best_overall_model(self) -> ModelConfiguration | None:
        """
        Return the highest scoring model based on ModelConfiguration sorting.
        """
        if not self.results:
            return None
        # Uses the __lt__ implementation from ModelConfiguration
        return max(self.results)

    def _calculate_row_counts(self) -> None:
        """Derive row counts and data loss metrics from current splits."""
        ds = self.data_splits
        self.train_row_count = len(ds.train_features)
        self.val_row_count = len(ds.val_features)
        self.test_row_count = len(ds.test_features)

        self.processed_row_count = self.train_row_count + self.val_row_count
        self.cleaned_row_count = self.processed_row_count + self.test_row_count

        self.dropped_row_count = ds.original_row_count - self.cleaned_row_count

        if ds.original_row_count > 0:
            self.row_loss_pct = (self.dropped_row_count / ds.original_row_count) * 100
        else:
            self.row_loss_pct = 0.0

    def get_state(self, include_preds: bool = False) -> dict[str, Any]:
        """
        Export the full audit snapshot as a serializable dictionary.

        Parameters
        ----------
        include_preds : bool, default False
            If True, converts large prediction arrays into lists for the export.

        Returns
        -------
        dict[str, Any]
            A nested dictionary of metadata, aggregate stats, and per-model results.
        """
        # Calculate aggregate compute metrics
        total_cpu = sum(res.total_duration for res in self.results)
        max_ram = max((res.actual_peak_gb for res in self.results), default=0.0)
        total_fits = sum(res.num_candidates * res.cv for res in self.results)

        # Calculate average efficiency (rows/sec) safely
        eff_list = [
            res.efficiency(data_splits=self.data_splits) for res in self.results
        ]
        valid_eff = [e for e in eff_list if e > 0]
        avg_efficiency = float(np.mean(valid_eff)) if valid_eff else 0.0

        return {
            "metadata": {
                "timestamp": self.audit_timestamp,
                "dataset_name": self.dataset_name,
                "duration_seconds": self.duration,
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
                "anomalies": {
                    "threshold": self.anomaly_threshold,
                    "display_map": self.anomaly_display_map,
                    "top_n": self.top_n_anomalies,
                    "dynamic_features": self.anomaly_dynamic_features,
                    "formats": {
                        "actual": self.actual_value_fmt.to_dict(),
                        "predicted": self.predicted_value_fmt.to_dict(),
                        "abs_error": self.abs_error_fmt.to_dict(),
                    },
                    # DataFrame conversion to list of records for JSON
                    "data": (
                        self.anomaly_data.to_dict(orient="records")
                        if self.anomaly_data is not None
                        else []
                    ),
                },
                "aggregate_stats": {
                    "total_cpu_time": total_cpu,
                    "peak_ram_gb": max_ram,
                    "total_cv_fits": total_fits,
                    "avg_rows_per_sec": avg_efficiency,
                },
                "model_limits": {
                    "accuracy_min": self.model_accuracy_limit,
                    "acceptable_min": self.model_acceptable_limit,
                    "stability_min": self.model_stability_limit,
                    "efficiency_min": self.model_efficiency_threshold,
                },
                "features": {k: f.to_dict() for k, f in self.features.items()},
            },
            "results": [
                res.to_dict(include_preds=include_preds) for res in self.results
            ],
        }

    def create_metadata(self) -> None:
        """
        Compute and finalize metadata describing the audit dataset and results.

        This method synchronizes row counts, loss percentages, and feature sets
        to ensure all reporting properties are accurate before export.
        """
        # 1. Update features-to-fit set in case metadata was added late
        self._init_features_to_fit_set()

        # 2. Recalculate row counts and data loss metrics
        # This handles the row_loss_pct assignment through the @setter logic
        self._calculate_row_counts()

        # 3. Guard against early calls before models have finished
        if len(self.results) == 0:
            return

    def capture_anomaly_context(self) -> None:
        """
        Identify anomalous rows and high-kurtosis features for audit context.

        Analyzes the best-performing model to extract anomalies and identifies
        features with the most extreme distributions to aid in error analysis.
        For regression, anomalies are rows whose absolute residual exceeds the
        ``anomaly_threshold`` percentile. For classification, anomalies are
        misclassified rows.
        """
        config = self.best_overall_model
        if config is None:
            return

        # 1. Calculate Absolute Errors
        y_val = self.data_splits.val_target
        preds_val = (
            config.preds_val
            if config.preds_val is not None
            else pd.Series(dtype="float64")
        )

        # Flatten and align to ensure matching indices.
        if config.task_type == TaskType.CLASSIFICATION:
            y_aligned = y_val.to_numpy().flatten().astype(str)
            p_aligned = preds_val.to_numpy().flatten().astype(str)
            abs_errors = (y_aligned != p_aligned).astype(np.float64)

            # For classification, anomalies are misclassifications.
            anomaly_mask = abs_errors > 0.0
        else:
            abs_errors = np.abs(
                y_val.to_numpy().flatten() - preds_val.to_numpy().flatten()
            )

            # 2. Extract Anomaly Mask
            # We use a percentile-based threshold (e.g., top 5% of errors).
            threshold = np.percentile(abs_errors, self.anomaly_threshold)
            anomaly_mask = abs_errors >= threshold

        # 3. Build Anomaly DataFrame (Inverting Scaling for Readability)
        anomalies_scaled = self.data_splits.val_features.iloc[anomaly_mask].copy()
        anomalies = self.data_splits.inverse_transform_df(anomalies_scaled)

        anomalies[AUDIT_ANOMALY_ACTUAL_COL] = y_val.iloc[anomaly_mask].values
        anomalies[AUDIT_ANOMALY_PREDICTED_COL] = preds_val.iloc[anomaly_mask].values
        anomalies[AUDIT_ANOMALY_ABS_ERROR_COL] = abs_errors[anomaly_mask]

        # 4. Identify High-Kurtosis Features (Dynamic Context)
        # Only numeric columns support kurtosis calculations
        numeric_df = self.data_splits.val_features.select_dtypes(include=[np.number])
        dynamic_features = (
            numeric_df.kurt()
            .sort_values(ascending=False)
            .index[: self.top_n_importance]
            .tolist()
        )

        # 5. Store Results Sorted by Error Magnitude
        self.anomaly_data = anomalies.sort_values(
            AUDIT_ANOMALY_ABS_ERROR_COL, ascending=False
        )
        self.anomaly_dynamic_features = dynamic_features

    def get_summary_data(
        self,
        config: ModelConfiguration,
        key_case: StringCase = StringCase.ORIGINAL,
    ) -> dict[str, Any]:
        """
        Build a flat summary record for a single model configuration.

        Parameters
        ----------
        config : ModelConfiguration
            The specific run to summarize.
        key_case : StringCase, default ORIGINAL
            The casing convention to apply to the dictionary keys.
        """
        # Ensure scores are non-null for the summary
        train_s = config.score_train or 0.0
        val_s = config.score_val or 0.0
        clean_s = config.score_val_cleaned or 0.0

        # Mapping key display labels
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
            "Duration",
            "Efficiency",
            "Train Score",
            "Gap",
            "Status",
            "Train Mean",
            "Train Std",
            "Train Median",
            "Train Skew",
            "Train Kurt",
            "Val Mean",
            "Val Std",
            "Val Median",
            "Val Skew",
            "Val Kurt",
            "Mean Delta",
            "Std Delta",
            "Quality Score",
            "Drift Index",
        ]

        # Apply casing transformation
        if key_case != StringCase.ORIGINAL:
            keys = convert_list_to_case(keys, key_case)
            conv_f = func_for_string_conv(key_case)
        else:
            conv_f = to_original_string

        values = [
            config.id,
            conv_f(config.model_type.value),
            conv_f(config.balancing_strategy.value),
            config.available_gb,
            config.estimated_peak_gb,
            config.actual_peak_gb,
            config.memory_risk_triggered,
            config.sampling_factor,
            config.concurrent_workers,
            val_s,
            clean_s,
            config.total_duration,
            config.efficiency(self.data_splits),
            train_s,
            config.gap,
            conv_f(config.model_generalization.value),
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

        return dict(zip(keys, values))

    def get_leaderboard(self) -> str:
        """Generate the console-friendly leaderboard summary."""
        state = self.get_state(include_preds=False)
        meta = state["metadata"]
        counts = meta["row_counts"]

        i_fmt = IntegerFormat()
        dur_fmt = DateTimeFormat(use_duration_format=True)
        ram_fmt = DataFormat(
            data_scale=DataScale.GB, precision=2, include_space_before_scale=True
        )

        duration_seconds = float(meta["duration_seconds"])
        peak_ram_val = float(meta["aggregate_stats"]["peak_ram_gb"])

        # `peak_ram_gb` is stored in GB; convert to bytes for DataFormat,
        # which handles scaling when configured with DataScale.GB.
        peak_ram_bytes = peak_ram_val * DataScale.GB.get_size()

        # 1. Build Metadata Header
        header_pairs = [
            ("Timestamp", meta["timestamp"]),
            ("Dataset", meta["dataset_name"]),
            (
                "Rows (T/V/Te)",
                f"{i_fmt.format_value(counts['train'])} / "
                f"{i_fmt.format_value(counts['val'])} / "
                f"{i_fmt.format_value(counts['test'])}",
            ),
            ("Duration", dur_fmt.format_value(duration_seconds)),
            ("Peak RAM", ram_fmt.format_value(peak_ram_bytes)),
        ]

        grid_pad = max((len(f.name) for f in self.features_used), default=10) + 4
        feat_grid = format_as_grid(
            input=[f.name for f in self.features_used],
            cols=3,
            padding=grid_pad,
            indent=4,
        )

        header = (
            f"\n{'=' * prefs.report_width}\n"
            f"AUDIT SNAPSHOT: {self.dataset_name}\n"
            f"{'-' * prefs.report_width}\n"
            f"{format_label_value_pairs(header_pairs, padding=4)}\n"
            f"Features Analyzed:\n{feat_grid}\n"
            f"{'-' * prefs.report_width}\n"
        )

        # 2. Build Results Table
        summary_list = [self.get_summary_data(cfg) for cfg in self.results]
        df = pd.DataFrame(summary_list).sort_values("Val Score", ascending=False)

        # Truncate columns for console display to prevent wrapping
        display_cols = [
            "ID",
            "Model",
            "Val Score",
            "Cleaned Score",
            "Gap",
            "Status",
            "Efficiency",
        ]
        return (
            header
            + df[display_cols].to_string(index=False)
            + f"\n{'=' * prefs.report_width}\n"
        )

    def get_best_by_type(self, model_type: ModelType) -> ModelConfiguration | None:
        """
        Retrieve the highest-scoring configuration for a specific model type.

        Parameters
        ----------
        model_type : ModelType
            The specific architecture to filter for (e.g., ModelType.RANDOM_FOREST).

        Returns
        -------
        ModelConfiguration | None
            The top-performing instance of that type, or None if not found.
        """
        # Filter results for the requested type
        typed_results = [r for r in self.results if r.model_type == model_type]

        if not typed_results:
            return None

        # Returns the max based on ModelConfiguration's __lt__ (score_val)
        return max(typed_results)

    def _extract_preds_and_probs(self) -> list[ModelPredictions]:
        """
        Detach prediction/probability arrays from results for memory-safe serialization.

        Returns
        -------
        list[ModelPredictions]
            A list of prediction containers aligned index-for-index with the results list.
        """
        new_results: list[ModelConfiguration] = []
        model_predictions: list[ModelAuditSummary.ModelPredictions] = []

        for config in self.results:
            # 1. Capture the arrays in the side-car container
            model_predictions.append(
                ModelAuditSummary.ModelPredictions(
                    preds_val=config.preds_val,
                    probs_val=config.probs_val,
                    preds_test=config.preds_test,
                    probs_test=config.probs_test,
                )
            )

            # 2. Clear the arrays from the main config to reduce object size
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
        self, model_predictions: list[ModelPredictions]
    ) -> None:
        """
        Reattach prediction/probability arrays to results after serialization tasks.

        Parameters
        ----------
        model_predictions : list[ModelPredictions]
            The list of prediction containers to re-merge into the results.
        """
        # Ensure we are iterating based on the current results length to maintain alignment
        new_results: list[ModelConfiguration] = []

        for index, config in enumerate(self.results):
            # We use index-based lookup to match the 'side-car' list back to the configs
            pred_set = model_predictions[index]

            new_results.append(
                dataclasses.replace(
                    config,
                    preds_val=pred_set.preds_val,
                    probs_val=pred_set.probs_val,
                    preds_test=pred_set.preds_test,
                    probs_test=pred_set.probs_test,
                )
            )

        self.results = new_results

    def export_results(
        self,
        prefix: str,
        file_type: FileType,
        path: PathLike,
        path_is_full_path: bool = False,
        append_timestamp_to_save_path: bool = False,
        report_title: str = "Model Audit Report",
    ) -> Path | CloudPath:
        """
        Export audit results to CSV, JSON, JOBLIB, Excel, or PDF.

        Parameters
        ----------
        prefix : str
            Filename prefix (e.g., "Audit_State").
        file_type : FileType
            A single FileType or bitmask of types to generate.
        path : PathLike
            Output directory or full file path.
        path_is_full_path : bool, default False
            If True, treats 'path' as the final destination file.
        append_timestamp_to_save_path : bool, default False
            Append audit timestamp to the output directory.
        report_title : str, default "Model Audit Report"
            The title applied to PDF/Excel report headers.

        Returns
        -------
        Path | CloudPath
            The path to the primary file generated.
        """
        from dsr_files.enums import FileType
        from dsr_files.json_handler import to_JSON_safe

        # 1. Resolve Pathing and Filenames
        if not path_is_full_path:
            output_dir = AnyPath(path)
            if append_timestamp_to_save_path:
                output_dir = output_dir / self.audit_timestamp
            filename = f"{prefix}_{self.audit_timestamp}"
        else:
            full_path_obj = AnyPath(path)
            output_dir = full_path_obj.parent
            filename = full_path_obj.stem

        output_dir.mkdir(parents=True, exist_ok=True)
        full_path = Path()

        # 2. Prepare Payload (Lightweight for JSON/CSV/Excel)
        export_payload = self.get_state(include_preds=False)
        metadata = export_payload["metadata"]
        anomalies = metadata["anomalies"]

        # Flattened metadata for tabular exports (CSV/Excel)
        meta_flat = {
            "audit_id": f"{prefix}_{self.audit_timestamp}",
            "timestamp": metadata["timestamp"],
            **metadata["row_counts"],
            **metadata["aggregate_stats"],
            "features_json": to_JSON_safe(metadata["features"]),
        }

        # --- CSV Export Logic ---
        if FileType.CSV in file_type:
            # Metadata Summary
            save_csv(
                pd.DataFrame(list(meta_flat.items())),
                output_dir,
                f"{filename}_metadata",
                header=False,
            )

            # Anomaly Context
            save_csv(
                pd.DataFrame([anomalies["display_map"]]),
                output_dir,
                f"{filename}_anomaly_map",
                header=False,
            )
            save_csv(
                pd.DataFrame(anomalies["data"]), output_dir, f"{filename}_anomaly_data"
            )
            save_csv(
                pd.DataFrame([anomalies["dynamic_features"]]),
                output_dir,
                f"{filename}_dynamic_features",
                header=False,
            )

            # Main Results (Leaderboard)
            full_path, _ = save_csv(
                pd.DataFrame(export_payload["results"]), output_dir, filename
            )

        # --- JSON Export Logic ---
        if FileType.JSON in file_type:
            full_path, _ = save_json(export_payload, output_dir, filename)

        # --- JOBLIB Export Logic (Full State) ---
        if FileType.JOBLIB in file_type:
            # Extract large arrays to optimize disk write, then restore
            preds_sidecar = self._extract_preds_and_probs()
            full_path, _ = save_joblib(self, output_dir, filename)
            self._restore_preds_and_probs(preds_sidecar)

        # --- EXCEL Export Logic ---
        if FileType.EXCEL in file_type:
            sheets = [
                ExcelSheetConfig(
                    pd.DataFrame(list(meta_flat.items())), "Audit Summary", header=False
                ),
                ExcelSheetConfig(
                    pd.DataFrame(export_payload["results"]), "Leaderboard"
                ),
                ExcelSheetConfig(pd.DataFrame(anomalies["data"]), "Anomaly Log"),
                ExcelSheetConfig(
                    pd.DataFrame(metadata["features"]).T, "Feature Metadata", index=True
                ),
            ]
            full_path, _ = save_excel(sheets, output_dir, filename)

        # --- PDF Export Logic ---
        if FileType.PDF in file_type:
            renderer = AuditPDFRenderer(summary=self, report_title=report_title)
            pdf_doc = renderer.render()
            full_path = pdf_doc.save(output_dir=output_dir, filename=filename)

        return full_path

    def evaluate_test_model(
        self, index: int, joblib_fullpath: PathLike | None = None
    ) -> None:
        """
        Evaluate a single model on the test set and optionally persist state.

        Parameters
        ----------
        index : int
            Index of the model configuration in the results list.
        joblib_fullpath : PathLike, optional
            Path to update the .joblib snapshot immediately after evaluation.
        """
        if index >= len(self.results):
            print(
                f"Error: Index {index} is out of bounds for results (len={len(self.results)})"
            )
            return

        config = self.results[index]
        print(
            f"Evaluating Test Set performance for {config.model_type.value} [ID: {config.id}]"
        )

        # 1. Hydrate the specific model architecture from its frozen config
        model = ModelSpecification.create_model_from_config(config)

        if model:
            # 2. Run test evaluation and update the results entry
            self.results[index] = model.evaluate_test_set_performance(
                data_splits=self.data_splits,
                config=config,
                features_to_fit_set=self.features_to_fit_set,
            )

            # 3. Optional Persistence
            if joblib_fullpath:
                from dsr_files.enums import FileType

                self.export_results(
                    prefix="Audit_State",
                    file_type=FileType.JOBLIB,
                    path=joblib_fullpath,
                    append_timestamp_to_save_path=False,
                    path_is_full_path=True,
                )
                print(
                    f"Audit snapshot updated on disk: {AnyPath(joblib_fullpath).name}"
                )
        else:
            print(
                f"Warning: Unable to instantiate {config.model_type.name} specification."
            )

    def evaluate_test_models(
        self, indexes: list[int], joblib_fullpath: PathLike | None = None
    ) -> None:
        """
        Evaluate a specific list of model indices on the test set.

        Parameters
        ----------
        indexes : list[int]
            List of result indices to process.
        joblib_fullpath : PathLike, optional
            Path to update the .joblib snapshot after each evaluation.
        """
        for index in indexes:
            self.evaluate_test_model(index=index, joblib_fullpath=joblib_fullpath)

    def evaluate_all_test_models(self, joblib_fullpath: PathLike | None = None) -> None:
        """
        Evaluate every model in the results list on the test set.

        Parameters
        ----------
        joblib_fullpath : PathLike, optional
            Path to update the .joblib snapshot during the process.
        """
        # Iterate over all indices in the current results list
        all_indices = list(range(len(self.results)))
        self.evaluate_test_models(indexes=all_indices, joblib_fullpath=joblib_fullpath)
