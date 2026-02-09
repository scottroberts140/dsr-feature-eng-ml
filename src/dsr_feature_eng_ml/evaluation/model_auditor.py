"""Audit runner for evaluating and tuning model specifications."""

from __future__ import annotations
from dsr_feature_eng_ml.enums import OptimizationStrategy
from dsr_feature_eng_ml.evaluation.schema import (
    ModelConfiguration,
    ModelAuditorConfig,
)
from dsr_feature_eng_ml.evaluation.model_audit_summary import ModelAuditSummary
from dsr_utils.formatting import (
    IntegerFormat,
    DateTimeFormat,
    EnumFormat,
)
from dsr_feature_eng_ml.preferences import prefs
from dsr_feature_eng_ml.evaluation.schema import FeatureMetadata
import time
import dataclasses
from typing import Optional
from pathlib import Path
import sys


class AuditLogger:
    """Simple stdout tee that writes audit logs to a file."""

    def __init__(self, file_path):
        """Open a log file and mirror stdout to it.

        Args:
            file_path: Path to the log file to write.
        """
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        """Write a message to both terminal and log file."""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """Flush both terminal and file buffers."""
        # Necessary for compatibility with sys.stdout
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close the log file."""
        self.log.close()


class ModelAuditor:
    """Orchestrates the evaluation and tuning of multiple model specifications."""

    # Class-level counter for experiment phases
    phase_number: int = 0

    def __init__(self, config: ModelAuditorConfig):
        """Initialize the auditor and its summary container.

        Args:
            config: Audit configuration including data splits and models.
        """
        self.config = config
        self.summary = ModelAuditSummary(
            data_splits=config.data_splits,
            dataset_name=config.dataset_name,
            top_n_importance=config.top_n_importance,
            original_row_count=config.data_splits.original_row_count,
            features=config.features,
            top_n_anomalies=config.top_n_anomalies,
            anomaly_display_map=config.anomaly_display_map,
            actual_value_fmt=config.actual_value_fmt,
            predicted_value_fmt=config.predicted_value_fmt,
            abs_error_fmt=config.abs_error_fmt,
            error_pct_fmt=config.error_pct_fmt,
            anomaly_threshold=config.anomaly_threshold,
        )

        if config.auto_increment_phase:
            ModelAuditor.phase_number += 1

        self.current_phase = ModelAuditor.phase_number
        self.features_to_fit_set: set[FeatureMetadata] = FeatureMetadata.dict_to_set(
            feature_dict=config.features, target_column=config.data_splits.target_column
        )

    def run_audit(
        self,
        optimize: bool = True,
        save_path: Optional[str | Path] = None,
        append_timestamp_to_save_path: bool = False,
        max_sample_size: Optional[int] = None,
        perform_memory_check: bool = True,
        filter_outliers: bool = False,
        outlier_count: int = prefs.default_worst_errors_n,
        efficiency_threshold: int = prefs.default_efficiency_threshold,
    ) -> None:
        """Execute the audit for all configured models.

        Args:
            optimize: If True, run hyperparameter tuning before final fit.
            save_path: Directory (or base path) for logs and snapshot exports.
            append_timestamp_to_save_path: If True, append audit timestamp to the save path.
            max_sample_size: Optional cap on sample size during tuning.
            perform_memory_check: If True, estimate memory risk during tuning.
            filter_outliers: If True, remove worst outliers for evaluation.
            outlier_count: Number of worst errors to treat as outliers.
            efficiency_threshold: Rows/sec threshold for efficiency scoring.
        """
        file_path = Path(save_path) if save_path is not None else Path("")

        if append_timestamp_to_save_path:
            file_path = file_path / self.summary.audit_timestamp

        file_path.mkdir(parents=True, exist_ok=True)
        log_name = f"audit_trace_{self.summary.audit_timestamp}.log"
        log_path = file_path / log_name

        # Start Redirection
        logger = AuditLogger(log_path)
        original_stdout = sys.stdout
        sys.stdout = logger

        try:
            start_time = time.perf_counter()
            step_description = f"Phase {self.current_phase}: {self.config.dataset_name}"
            i = 0
            duration_format = DateTimeFormat(use_duration_format=True)
            model_type_format = EnumFormat(use_value=False)
            model_balancing_format = EnumFormat.from_format(model_type_format)
            id_format = IntegerFormat(width=2, pad_value="0")

            for model in self.config.models_to_run:
                print(
                    f"Auditing {model_type_format.format_value(model.model_type)} with {model_balancing_format.format_value(model.balancing_strategy)}... ",
                    end="",
                )

                # Global ID maps directly to the loop/leaderboard index
                i += 1
                global_id = f"{id_format.format_value(i)}"
                tuning_duration = 0.0
                best_cv_score: Optional[float] = None
                print_end = "" if prefs.cv_verbose == 0 else "\n"
                memory_risk_triggered: bool = False
                estimated_peak_gb: float = 0.0
                available_gb: float = 0.0
                model_multiplier: float = 1.0
                sampling_factor: float = 1.0

                if optimize:
                    # We use combined data (Train+Val) for the final CV search
                    print("Tune", end=print_end)
                    use_combined_data = True
                    tuning_start_time = time.perf_counter()
                    (
                        _,
                        best_cv_score,
                        memory_risk_triggered,
                        available_gb,
                        estimated_peak_gb,
                        model_multiplier,
                        sampling_factor,
                    ) = model.tune_model(
                        data_splits=self.summary.data_splits,
                        method=OptimizationStrategy.RANDOM_SEARCH,
                        features_to_fit_set=self.features_to_fit_set,
                        custom_grid=dataclasses.asdict(model.model_dials),
                        use_combined_data=use_combined_data,
                        max_sample_size=max_sample_size,
                        perform_memory_check=perform_memory_check,
                    )
                    tuning_end_time = time.perf_counter()
                    tuning_duration = tuning_end_time - tuning_start_time

                    if prefs.cv_verbose == 0:
                        print(
                            f" ({duration_format.format_value(tuning_duration)}) ",
                            end="",
                        )
                else:
                    use_combined_data = False

                # Generate the "Salami Slice" result
                print(
                    f"Final fit with score_cv={prefs.score_format.format_value(best_cv_score)}",
                    end=print_end,
                )
                fit_start_time = time.perf_counter()
                result: ModelConfiguration = model.fit_and_evaluate_val(
                    data_splits=self.summary.data_splits,
                    id=global_id,
                    features_to_fit_set=self.features_to_fit_set,
                    score_cv=best_cv_score,
                    use_combined_data=use_combined_data,
                    filter_outliers=filter_outliers,
                    outlier_count=outlier_count,
                )
                fit_end_time = time.perf_counter()
                fit_duration = fit_end_time - fit_start_time

                if prefs.cv_verbose == 0:
                    print(f" ({duration_format.format_value(fit_duration)}) ", end="")

                result = dataclasses.replace(
                    result,
                    id=global_id,
                    tuning_duration=tuning_duration,
                    fit_duration=fit_duration,
                    available_gb=available_gb,
                    estimated_peak_gb=estimated_peak_gb,
                    memory_risk_triggered=memory_risk_triggered,
                    sampling_factor=sampling_factor,
                    model_multiplier=model_multiplier,
                    concurrent_workers=model.n_jobs,
                    efficiency_threshold=efficiency_threshold,
                )

                if prefs.cv_verbose == 0:
                    print(
                        f"Done ({duration_format.format_value(tuning_duration + fit_duration)}) "
                    )

                # Track the result in the summary
                self.summary.add_model_configuration(result)

            end_time = time.perf_counter()
            self.summary.duration = end_time - start_time

            # Finalize the phase report
            self._report_phase_completion(step_description)
            self.summary.capture_anomaly_context()
            from dsr_files.enums import FileType

            save_path = self.summary.export_results(
                prefix="Audit_State",
                file_type=FileType.JOBLIB,
                path=file_path,
                append_timestamp_to_save_path=False,
            )
            print(f"Full Audit Snapshot saved: {save_path}")
        finally:
            print(f"--- Audit Trace Saved to {log_name} ---")
            sys.stdout = original_stdout
            logger.close()

    def _report_phase_completion(self, step_description: str):
        """Finalize a phase by capturing metadata and printing the leaderboard."""
        self.summary.create_metadata()
        print(f"\n--- {step_description} Completed ---")

        # Wrap in print() to display the DataFrame returned
        print(self.summary.get_leaderboard())

        self._update_model_viability()

    def _update_model_viability(self):
        """
        Filters the list of models for the next run based on the viable_score_gap.
        Models that fall too far behind the current leader are marked as non-viable.
        """
        best_config = self.summary.best_overall_model
        # Guard against no results or a None score in the leader
        if not best_config or best_config.score_val is None:
            return

        leader_score = best_config.score_val
        viable_models = []

        for model in self.config.models_to_run:
            model_result = self.summary.get_best_by_type(model.model_type)

            # Check if the model result exists AND has a valid score
            if model_result and model_result.score_val is not None:
                gap_from_leader = leader_score - model_result.score_val

                if gap_from_leader < self.config.viable_score_gap:
                    viable_models.append(model)
            else:
                # If a model hasn't run or failed to produce a score,
                # you might want to keep it by default or drop it.
                # Keeping it is usually safer:
                viable_models.append(model)

        self.config.models_to_run = viable_models
