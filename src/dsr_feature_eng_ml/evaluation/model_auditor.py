"""Audit runner for evaluating and tuning model specifications."""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

from cloudpathlib import AnyPath
from dsr_files.utils import PathLike
from dsr_utils.formatting import DataScale, DateTimeFormat, EnumFormat, IntegerFormat

from dsr_feature_eng_ml.enums import OptimizationStrategy
from dsr_feature_eng_ml.evaluation.model_audit_summary import ModelAuditSummary
from dsr_feature_eng_ml.evaluation.schema import FeatureMetadata, ModelAuditorConfig
from dsr_feature_eng_ml.prefs_instance import prefs


class AuditLogger:
    """
    Standard output 'Tee' that mirrors terminal output to a log file.

    This class captures all print statements and redirects them to both the
    console and a specified file, providing a persistent record of the audit.
    """

    def __init__(self, file_path: PathLike):
        """
        Initialize the logger and open the target log file.

        Parameters
        ----------
        file_path : str | Path | CloudPath
            The destination path for the audit log file.
        """
        self.terminal = sys.stdout
        # We use 'w' to overwrite previous audits; change to 'a' if appending is preferred.
        self.log = AnyPath(file_path).open("w", encoding="utf-8")

    def write(self, message: str) -> None:
        """Write a message to both the terminal and the log file."""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        """
        Flush both terminal and file buffers.

        This ensures logs are written immediately rather than waiting for
        internal buffer limits.
        """
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        """Close the log file and release resources."""
        if not self.log.closed:
            self.log.close()

    def __enter__(self) -> "AuditLogger":
        """Enable context manager support."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Ensure the file is closed upon exiting the context."""
        self.close()


class ModelAuditor:
    """
    Orchestrates the evaluation and tuning of multiple model specifications.

    Manages the lifecycle of an experiment phase, including tuning, final fitting,
    memory telemetry, and the persistence of audit traces.
    """

    # Global tracking for experiment iterations
    phase_number: int = 0

    def __init__(self, config: ModelAuditorConfig):
        """Initialize the auditor and its summary container."""
        self.config = config

        # Initialize the summary with diagnostic parameters from config
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

        # Filter features eligible for training and align with transformed columns.
        self.features_to_fit_set = self._resolve_features_to_fit_set()

    def _resolve_features_to_fit_set(self) -> set[FeatureMetadata]:
        """Resolve fit features against transformed split columns.

        If a configured feature was one-hot encoded, this expands it into its
        dummy columns and keeps a parent_name link to the original feature.
        """
        target_column = self.config.data_splits.target_column
        available_columns = self.config.data_splits.train_features.columns
        available_set = set(available_columns)

        resolved: set[FeatureMetadata] = set()

        for feature in self.config.features.values():
            if not feature.is_used_in_fit or feature.name == target_column:
                continue

            if feature.name in available_set:
                resolved.add(feature)
                continue

            encoded_columns = sorted(
                col for col in available_columns if col.startswith(f"{feature.name}_")
            )
            for i, encoded_col in enumerate(encoded_columns, start=1):
                resolved.add(
                    FeatureMetadata(
                        name=encoded_col,
                        id=f"{feature.id}_{i:02d}",
                        position=available_columns.tolist().index(encoded_col),
                        short_name=feature.short_name,
                        formatter=feature.formatter,
                        description=feature.description,
                        is_used_in_fit=True,
                        parent_name=feature.name,
                    )
                )

        if resolved:
            return resolved

        return FeatureMetadata.dict_to_set(
            feature_dict=self.config.features,
            target_column=target_column,
        )

    def run_audit(
        self,
        optimize: bool = True,
        save_path: PathLike | None = None,
        append_timestamp_to_save_path: bool = False,
        max_sample_size: int | None = None,
        perform_memory_check: bool = True,
        filter_outliers: bool = False,
        outlier_count: int = prefs.default_worst_errors_n,
        efficiency_threshold: int = prefs.default_efficiency_threshold,
    ) -> None:
        """
        Execute the audit for all configured models.

        Parameters
        ----------
        optimize : bool, default True
            Run hyperparameter tuning before the final fit.
        save_path : str | Path | CloudPath, optional
            Base directory for logs and exports.
        max_sample_size : int, optional
            Cap on training rows used during the tuning phase.
        perform_memory_check : bool, default True
            Calculate memory risk/headroom before tuning.
        """
        base_dir = AnyPath(save_path) if save_path else Path.cwd()

        if append_timestamp_to_save_path:
            base_dir = base_dir / self.summary.audit_timestamp

        base_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"audit_trace_{self.summary.audit_timestamp}.log"
        log_path = base_dir / log_name

        # Standard output redirection using AuditLogger
        original_stdout = sys.stdout

        # Use context manager to ensure logger closure
        with AuditLogger(log_path) as logger:
            sys.stdout = logger
            try:
                start_perf = time.perf_counter()
                step_desc = f"Phase {self.current_phase}: {self.config.dataset_name}"

                # Formatters for cleaner console output
                dur_fmt = DateTimeFormat(use_duration_format=True)
                type_fmt = EnumFormat(use_value=False)
                bal_fmt = EnumFormat.from_format(type_fmt)
                id_fmt = IntegerFormat(width=2, pad_value="0")

                for i, model in enumerate(self.config.models_to_run, 1):
                    print(
                        f"Auditing {type_fmt.format_value(model.model_type)} "
                        f"[{bal_fmt.format_value(model.balancing_strategy)}]... ",
                        end="",
                    )

                    global_id = id_fmt.format_value(i)
                    tuning_dur = 0.0
                    best_cv: float | None = None
                    print_end = "\n" if prefs.cv_verbose > 0 else ""

                    # 1. Tuning Phase
                    if optimize:
                        print("Tuning", end=print_end)
                        t_start = time.perf_counter()
                        (
                            _,
                            best_cv,
                            mem_risk,
                            avail_gb,
                            est_peak,
                            mult,
                            samp_factor,
                        ) = model.tune_model(
                            data_splits=self.summary.data_splits,
                            method=OptimizationStrategy.RANDOM_SEARCH,
                            features_to_fit_set=self.features_to_fit_set,
                            custom_grid=dataclasses.asdict(model.model_dials),
                            use_combined_data=True,
                            max_sample_size=max_sample_size,
                            perform_memory_check=perform_memory_check,
                        )
                        tuning_dur = time.perf_counter() - t_start
                        if prefs.cv_verbose == 0:
                            print(f" ({dur_fmt.format_value(tuning_dur)}) ", end="")
                    else:
                        mem_risk, avail_gb, est_peak, mult, samp_factor = (
                            False,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                        )

                    # 2. Final Fit and Evaluation
                    print(
                        f"Fitting (CV={prefs.score_format.format_value(best_cv)})",
                        end=print_end,
                    )
                    f_start = time.perf_counter()
                    result = model.fit_and_evaluate_val(
                        data_splits=self.summary.data_splits,
                        id=global_id,
                        features_to_fit_set=self.features_to_fit_set,
                        score_cv=best_cv,
                        use_combined_data=optimize,  # Logic matches tune phase
                        filter_outliers=filter_outliers,
                        outlier_count=outlier_count,
                    )
                    fit_dur = time.perf_counter() - f_start

                    # 3. Post-Process Results
                    result = dataclasses.replace(
                        result,
                        tuning_duration=tuning_dur,
                        fit_duration=fit_dur,
                        available_gb=avail_gb,
                        estimated_peak_gb=DataScale.GB.get_scaled_value(est_peak),
                        memory_risk_triggered=mem_risk,
                        sampling_factor=samp_factor,
                        model_multiplier=mult,
                        concurrent_workers=model.n_jobs,
                        efficiency_threshold=efficiency_threshold,
                    )

                    if prefs.cv_verbose == 0:
                        total_m_dur = tuning_dur + fit_dur
                        print(f"Done ({dur_fmt.format_value(total_m_dur)})")

                    self.summary.add_model_configuration(result)

                # Finalize summary
                self.summary.duration = time.perf_counter() - start_perf
                self._report_phase_completion(step_desc)
                self.summary.capture_anomaly_context()

                # Export snapshot
                from dsr_files.enums import FileType

                exp_path = self.summary.export_results(
                    prefix="Audit_State",
                    file_type=FileType.JOBLIB,
                    path=base_dir,
                    append_timestamp_to_save_path=False,
                )
                print(f"Audit Snapshot: {exp_path}")

            except Exception as e:
                print(f"CRITICAL AUDIT ERROR: {str(e)}")
                raise e
            finally:
                sys.stdout = original_stdout

        print(f"--- Audit Trace Saved: {log_name} ---")

    def _report_phase_completion(self, step_description: str) -> None:
        """Finalize phase metadata and display results."""
        self.summary.create_metadata()
        print(f"\n--- {step_description} Completed ---")
        print(self.summary.get_leaderboard())
        self._update_model_viability()

    def _update_model_viability(self) -> None:
        """Prune non-performing models from future phases based on lead gap."""
        best_cfg = self.summary.best_overall_model
        if not best_cfg or best_cfg.score_val is None:
            return

        leader_score = best_cfg.score_val
        viable_models = []

        for model in self.config.models_to_run:
            m_res = self.summary.get_best_by_type(model.model_type)

            if m_res and m_res.score_val is not None:
                gap = leader_score - m_res.score_val
                if gap < self.config.viable_score_gap:
                    viable_models.append(model)
            else:
                # Keep models that haven't run yet or didn't score
                viable_models.append(model)

        self.config.models_to_run = viable_models
