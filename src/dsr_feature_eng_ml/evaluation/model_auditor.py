"""Audit runner for evaluating and tuning model specifications."""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from pathlib import Path
from types import TracebackType

from cloudpathlib import AnyPath
from dsr_files.utils import PathLike
from dsr_utils.formatting import (
    DataScale,
    DateTimeFormat,
    EnumFormat,
    IntegerFormat,
    NumericScale,
)

from dsr_feature_eng_ml.enums import OptimizationStrategy
from dsr_feature_eng_ml.evaluation.model_audit_summary import ModelAuditSummary
from dsr_feature_eng_ml.evaluation.schema import FeatureMetadata, ModelAuditorConfig
from dsr_feature_eng_ml.prefs_instance import prefs


class AuditLogger:
    """
    Standard output 'Tee' that mirrors terminal output to a log file, and also
    installs a :class:`logging.FileHandler` so that Python ``logging`` calls at
    or above ``log_level`` are captured in the same file.

    On entry:

    - Opens the log file (truncating any previous content).
    - Redirects ``sys.stdout`` writes to both the terminal and the log file.
    - Adds a ``logging.FileHandler`` to the root logger so that
      ``logging.info()``, ``logging.warning()``, etc. are written to the file.

    On exit (or ``close()``):

    - Removes the ``FileHandler`` from the root logger and closes it.
    - Closes the tee file handle.

    The caller is responsible for restoring ``sys.stdout`` (typically via a
    ``try/finally`` around ``sys.stdout = logger``).
    """

    _LOG_FORMATTER = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    def __init__(self, file_path: PathLike, log_level: int | str = logging.INFO):
        """
        Initialize the logger and open the target log file.

        Parameters
        ----------
        file_path : PathLike
            The destination path for the audit log file.
        log_level : int | str, default ``logging.INFO``
            Minimum Python logging level written to the log file. Accepts
            integer constants (e.g. ``logging.DEBUG``) or level-name strings
            (e.g. ``"INFO"``, ``"WARNING"``).
        """
        # Resolve the numeric level first so validation happens before any I/O.
        if isinstance(log_level, str):
            numeric_level = logging.getLevelName(log_level.upper())
            if not isinstance(numeric_level, int):
                raise ValueError(
                    f"Invalid log_level: {log_level!r}. "
                    "Use a standard Python logging level name such as "
                    "'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'."
                )
        else:
            numeric_level = log_level

        self.terminal = sys.stdout

        # Open file in "w" mode to start a fresh log for this audit session.
        self.log = AnyPath(file_path).open("w", encoding="utf-8")

        # Install a FileHandler on the root logger so that logging.* calls are
        # also captured. Use "a" so it appends after the tee already created
        # (and truncated) the file above.
        self._log_handler = logging.FileHandler(
            str(Path(str(file_path)).resolve()), mode="a", encoding="utf-8"
        )
        self._log_handler.setLevel(numeric_level)
        self._log_handler.setFormatter(self._LOG_FORMATTER)

        root = logging.getLogger()
        self._original_root_level = root.level
        # Ensure the root logger doesn't silently drop records below our level.
        if root.level == logging.NOTSET or root.level > numeric_level:
            root.setLevel(numeric_level)
        root.addHandler(self._log_handler)

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
        if not self.log.closed:
            self.log.flush()

    def close(self) -> None:
        """Remove the logging handler, close the log file, and release resources."""
        root = logging.getLogger()
        root.removeHandler(self._log_handler)
        self._log_handler.close()
        root.setLevel(self._original_root_level)
        if not self.log.closed:
            self.log.close()

    def __enter__(self) -> "AuditLogger":
        """Enable context manager support."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
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
            count_numeric_scale=config.count_numeric_scale,
            pdf_feature_importance_chart_limit=config.pdf_feature_importance_chart_limit,
            anomaly_table_max_columns=config.anomaly_table_max_columns,
            anomaly_table_show_notes=config.anomaly_table_show_notes,
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
        count_numeric_scale: NumericScale | None = None,
        max_sample_size: int | None = None,
        perform_memory_check: bool = True,
        filter_outliers: bool = False,
        outlier_count: int = prefs.default_worst_errors_n,
        efficiency_threshold: int = prefs.default_efficiency_threshold,
        audit_log_level: str = "INFO",
    ) -> None:
        """
        Execute the audit for all configured models.

        Parameters
        ----------
        optimize : bool, default True
            Run hyperparameter tuning before the final fit.
        save_path : PathLike, optional
            Base directory for logs and exports.
        append_timestamp_to_save_path : bool, default False
            If True, write artifacts into a timestamped subdirectory under
            ``save_path`` using the audit summary timestamp.
        max_sample_size : int, optional
            Cap on training rows used during the tuning phase.
        perform_memory_check : bool, default True
            Calculate memory risk/headroom before tuning.
        filter_outliers : bool, default False
            If True, compute additional cleaned validation metrics by removing
            the highest-error observations from the validation set.
        outlier_count : int, default prefs.default_worst_errors_n
            Maximum number of high-error validation observations to exclude
            when ``filter_outliers`` is enabled. A safety cap inside the
            scoring pipeline prevents removing more than half of the samples.
        efficiency_threshold : int, default prefs.default_efficiency_threshold
            Minimum throughput threshold, in rows per second, used when storing
            evaluation metadata and downstream recommendation signals.
        audit_log_level : str, default ``"INFO"``
            Minimum Python logging level written to the audit log file. Stdout
            (print) output is always tee'd to both terminal and file regardless
            of this setting. Accepts standard level names such as ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``.
        """
        self.summary.count_numeric_scale = (
            self.config.count_numeric_scale
            if count_numeric_scale is None
            else count_numeric_scale
        )

        base_dir = AnyPath(save_path) if save_path else Path.cwd()

        if append_timestamp_to_save_path:
            base_dir = base_dir / self.summary.audit_timestamp

        base_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"audit_trace_{self.summary.audit_timestamp}.log"
        log_path = base_dir / log_name

        # Standard output redirection using AuditLogger
        original_stdout = sys.stdout

        # Use context manager to ensure logger closure
        with AuditLogger(log_path, log_level=audit_log_level) as logger:
            sys.stdout = logger
            try:
                start_perf = time.perf_counter()
                step_desc = f"Phase {self.current_phase}: {self.config.dataset_name}"

                # Formatters for cleaner console output
                dur_fmt = DateTimeFormat(use_duration_format=True)
                type_fmt = EnumFormat(use_value=False)
                bal_fmt = EnumFormat.from_format(type_fmt)
                id_fmt = IntegerFormat(width=2, pad_value="0")
                total_models = len(self.config.models_to_run)
                fitted_estimators: dict[str, object] = {}

                for i, model in enumerate(self.config.models_to_run, 1):
                    pending_tune_complete: str | None = None

                    def emit_stage_progress(message: str) -> None:
                        nonlocal pending_tune_complete
                        if message.startswith("tune: search complete"):
                            pending_tune_complete = message
                            return

                        print(f"  {message}", flush=True)

                    print(
                        f"Auditing {type_fmt.format_value(model.model_type)} "
                        f"[{bal_fmt.format_value(model.balancing_strategy)}]... ",
                        end="",
                        flush=True,
                    )

                    global_id = id_fmt.format_value(i)
                    tuning_dur = 0.0
                    best_cv: float | None = None

                    # 1. Tuning Phase
                    if optimize:
                        print("Tuning", flush=True)
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
                            progress_callback=emit_stage_progress,
                        )
                        tuning_dur = time.perf_counter() - t_start
                        if prefs.cv_verbose == 0:
                            if pending_tune_complete is not None:
                                print(
                                    "  "
                                    f"{pending_tune_complete} "
                                    f"({dur_fmt.format_value(tuning_dur)})",
                                    flush=True,
                                )
                                pending_tune_complete = None
                    else:
                        mem_risk, avail_gb, est_peak, mult, samp_factor = (
                            False,
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                        )

                    # 2. Final Fit and Evaluation
                    cv_display = (
                        prefs.score_format.format_value(best_cv)
                        if best_cv is not None
                        else "N/A"
                    )
                    print(f"Fitting (CV={cv_display})", flush=True)
                    f_start = time.perf_counter()
                    result = model.fit_and_evaluate_val(
                        data_splits=self.summary.data_splits,
                        id=global_id,
                        features_to_fit_set=self.features_to_fit_set,
                        score_cv=best_cv,
                        use_combined_data=False,
                        filter_outliers=filter_outliers,
                        outlier_count=outlier_count,
                        progress_callback=emit_stage_progress,
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
                        print(
                            f"  Done ({dur_fmt.format_value(total_m_dur)})",
                            flush=True,
                        )

                    self.summary.add_model_configuration(result)
                    if model.estimator is None:
                        raise RuntimeError(
                            "Estimator unexpectedly missing after fit for model "
                            f"id {global_id}."
                        )
                    fitted_estimators[global_id] = model.estimator
                    print(
                        "Progress: "
                        f"{i}/{total_models} models recorded "
                        f"(memory_risk={result.memory_risk_triggered}, "
                        f"workers={result.concurrent_workers})"
                    )

                # Finalize summary
                print("Finalizing audit metadata...")
                self.summary.duration = time.perf_counter() - start_perf
                self._report_phase_completion(step_desc)
                print("Capturing anomaly context...")
                self.summary.capture_anomaly_context()
                print("Writing audit snapshot (.joblib)...")

                # Export snapshot
                from dsr_files.enums import FileType

                exp_path = self.summary.export_results(
                    prefix="Audit_State",
                    file_type=FileType.JOBLIB,
                    path=base_dir,
                    append_timestamp_to_save_path=False,
                )
                print(f"Audit Snapshot: {exp_path}")

                print("Writing fitted model artifacts (.joblib bundles + manifest)...")
                model_manifest_path = self.summary.export_results(
                    prefix="Audit_State",
                    file_type=FileType.MODEL,
                    path=base_dir,
                    append_timestamp_to_save_path=False,
                    fitted_models=fitted_estimators,
                )
                print(f"Model Artifact Manifest: {model_manifest_path}")

            except Exception as e:
                print(f"CRITICAL AUDIT ERROR: {str(e)}")
                raise
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
