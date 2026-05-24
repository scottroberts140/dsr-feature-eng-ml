# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.11] - 2026-05-24

### Added

* **Direct Fitted-Model Artifact Export**: Added `FileType.MODEL` export support in `ModelAuditSummary.export_results(...)` to persist per-model fitted estimator bundles (`.joblib`) plus a JSON manifest for downstream inference workflows.
* **Run-Audit Estimator Handoff**: `ModelAuditor.run_audit(...)` now captures each fitted estimator during the audit loop and triggers a dedicated model-artifact export phase after snapshot persistence.

### Changed

* **Model Artifact Folder Layout**: MODEL exports are now written under a `models/` subfolder beneath the selected output directory to keep root audit folders clean and predictable.

### Fixed

* **Auditor Test Output Isolation**: Updated auditor viability-pruning tests to pass `tmp_path` into `run_audit(...)`, preventing generated `Audit_State_*` and `audit_trace_*` artifacts from being written into the repository root during test runs.

## [1.3.10] - 2026-05-23

### Fixed

* **Lazy Export Symbol Declarations**: Added `TYPE_CHECKING` declarations in `__init__.py` for lazy-exported names (`validate_n_jobs`, `check_memory_risk`, and `AuditPDFRenderer`) so `__all__` symbols resolve cleanly in static analysis without changing runtime lazy import behavior.

## [1.3.9] - 2026-05-22

### Fixed

* **Snapshot Timestamp Normalization on Load**: `ModelAuditSummary.from_joblib(...)` now normalizes `audit_timestamp` from `Audit_State_<timestamp>.joblib` filenames when available, ensuring loaded snapshot metadata matches the selected artifact and downstream reporting paths.

## [1.3.8] - 2026-05-22

### Changed

* **Schema Typing Compatibility Cleanup**: Updated pandas Index conversions in `evaluation/schema.py` from `.tolist()` to `.to_list()` for improved static-analysis compatibility.

### Fixed

* **Pylance Recursive Series Diagnostics**: Added a module-level Pyright directive in `evaluation/schema.py` to suppress known false-positive recursive pandas `Series` typing errors that cascaded into spurious attribute diagnostics.

## [1.3.7] - 2026-05-20

### Added

* **Split-First Config Factory**: Added `ModelAuditorConfig.from_splits(...)` to build auditor configuration and instantiated models directly from pre-built `DataSplits`.

### Changed

* **Factory Consolidation**: Refactored `ModelAuditorConfig.from_dataset(...)` to construct `DataSplits` and delegate model/config assembly to `from_splits(...)`, reducing duplicated setup paths.

## [1.3.6] - 2026-05-17

### Added

* **XGBoost gamma Hyperparameter Support**: Added `gamma` (minimum loss reduction required to make a further partition) to `XGBClassifierParams`, estimator construction, parameter display, and search-grid options — quick grid uses `[0.0, 0.1]`, standard grid uses `[0.0, 0.1, 0.2]`.

## [1.3.5] - 2026-05-14

### Changed

* **README Example Formatting Cleanup**: Normalized the quick-start and preferences examples to the repository's current formatting style for easier copy-paste use.
* **Preferences Style Block Formatting Cleanup**: Reflowed the Matplotlib `rcParams` style map in `Preferences.apply_style()` without changing runtime behavior.

## [1.3.4] - 2026-05-14

### Added

* **XGBoost min_child_weight Hyperparameter Support**: Added `min_child_weight` to `XGBClassifierParams`, estimator construction, and expanded standard search-grid options so the parameter is tunable and configurable in classification workflows.

### Changed

* **Hyperparameter Display Labels**: Added a compact display alias for `min_child_weight` in preferences-backed reporting labels to keep audit views consistent with other XGBoost dial names.

## [1.3.3] - 2026-05-09

### Added

* **Data Profile PDF Page**: Added a dedicated Data Profile page to the audit PDF that surfaces per-split target distribution, class percentages, and majority-to-minority imbalance ratios.
* **Configurable Count Scaling in Audit Summaries**: Added `count_numeric_scale` to `ModelAuditorConfig` and `ModelAuditSummary` so row counts in PDF and exported audit metadata can be rendered with explicit numeric scale control.

### Changed

* **PDF Count Formatting Source of Truth**: Audit PDF row-count formatting now reads from `ModelAuditSummary.count_numeric_scale` instead of hard-coded renderer defaults, keeping exported presentation consistent with orchestrator settings.
* **Data Profile Table Layout**: Right-aligned count, percent, and ratio columns and simplified row grouping to improve scanability in the new class-balance table.

### Fixed

* **Data Profile Font Shrink Regression**: Restored consistent numeric alignment in the Data Profile table by relying on the updated `dsr-utils` overflow-only shrink-to-fit behavior.

## [1.3.2] - 2026-05-07

### Added

* **Task-Specific Metrics Display in PDF Reports**: Extended Detailed Audit Stats page to dynamically display task-specific performance metrics (3+ metrics per task type) in the rendered PDF table. Metrics are conditionally added based on audit `task_type`:
  * **Regression**: MAE, MSE, R² (in addition to CV Score, Val Score, Test Score)
  * **Classification**: Accuracy, ROC-AUC (in addition to CV Score, Val Score, Test Score)
  * All metrics formatted consistently with `prefs.score_format`.

### Fixed

* **Classification PDF Generation**: Added safety guard in `_plot_efficiency_scatter()` to prevent `KeyError` when MAE column is absent (regression-only metric). Classification tasks now default to uniform bubble size when MAE is unavailable.
* **Tuple Unpacking in Metric Collection**: Corrected unpacking of `_score_classification()` return values from 4 to 6 elements in `_get_validation_metrics()` and `_get_test_metrics()` methods in `ModelSpecification`. Returns now correctly capture ROC-AUC metrics: `(f1, f1_cleaned, roc_auc, roc_auc_cleaned, preds, probs)`.

## [1.3.1] - 2026-05-06

### Changed

* **Hyperparameter Label Readability**: Expanded and shortened hyperparameter display-name mappings to keep deep-dive configuration labels compact and consistent across model families.
* **Deep-Dive HP Table Layout Tuning**: Refined HP/Value column proportion defaults and spacing in the configuration table renderer to better preserve side-by-side multi-page readability.

### Fixed

* **Restored-Snapshot Residual Plots**: PDF export now hydrates missing validation prediction artifacts before rendering, preventing deep-dive residual panels from showing "No validation predictions available" after loading joblib snapshots.

## [1.3.0] - 2026-05-05

### Added

* **Classification ROC-AUC Metrics**: Added native ROC-AUC computation in classification scoring flows, including binary (`probs[:, 1]`) and multiclass (`ovr`, `weighted`) support.
* **Model Configuration ROC-AUC Fields**: Extended `ModelConfiguration` with `roc_auc_train`, `roc_auc_val`, `roc_auc_val_cleaned`, and `roc_auc_test` fields, with serialization support.
* **ROC-AUC Test Suite**: Added a dedicated test module for ROC-AUC behavior covering direct computation, schema field presence, multiclass variants, and edge cases.

### Changed

* **Validation/Test Metric Wiring**: Updated classification validation and test metric assembly to propagate computed ROC-AUC values through output metric dictionaries.
* **Test Module Cleanup**: Refined ROC-AUC test formatting and imports after initial suite addition.

## [1.2.4] - 2026-05-05

### Added

* **Task-Specialized ModelSpecification Bases**: Introduced `ClassificationModelSpecification` and `RegressionModelSpecification` to centralize task-specific scoring behavior and reduce branching in shared model orchestration logic.
* **Automatic Categorical Encoding Trace**: Added audit log visibility when `DataSplits.from_data_source` auto-applies one-hot encoding, including the affected source columns.
* **`AuditLogger` Python Logging Integration**: Extended `AuditLogger` to install a `logging.FileHandler` on the root logger on context entry, capturing `logging.info()`, `logging.warning()`, and other Python logging calls into the same log file as stdout. The tee behavior (stdout to both terminal and file) is unchanged. The `log_level` parameter (default `logging.INFO`) controls the minimum level written via the handler. `run_audit` exposes this as `audit_log_level: str = "INFO"`.
* **Live Audit Run Progress Output**: `ModelAuditor.run_audit` now emits stage-by-stage progress to the terminal throughout each model evaluation. Each model prints a `Tuning` and `Fitting` header (with forced flush) followed by indented sub-stage messages. `tune_model` emits checkpoints for data preparation, row sampling, memory checks, and search execution (including search type, candidates/params, CV folds, workers, and duration). `fit_and_evaluate_val` emits fit start, training completion, and validation completion. After all models complete, finalization messages indicate metadata assembly, anomaly context capture, and snapshot write, followed by the snapshot path. A per-model progress summary (`Progress: N/M models recorded`) is also printed after each model.
* **`progress_callback` Parameter on `tune_model` and `fit_and_evaluate_val`**: Both methods now accept an optional `progress_callback: Callable[[str], None] | None = None` parameter. When provided, the callback is invoked at each stage checkpoint, enabling callers to surface live terminal output. `run_audit` wires up callbacks automatically.
* **Separate Fit-Time Verbosity Preference**: Added `prefs.fit_verbose` as a dedicated non-negative integer preference for estimator fit-time verbosity. This is intentionally separate from `prefs.cv_verbose`, which remains scoped to CV searcher verbosity (`GridSearchCV`/`RandomizedSearchCV`).
* **Anomaly Table Column Cap**: Added `ModelAuditorConfig.anomaly_table_max_columns` and matching `ModelAuditSummary` support to optionally cap the number of dynamic context columns rendered on the Data Anomaly Log page.

### Changed

* **Task-Specific ModelParams Bases**: Introduced `ClassificationModelParams` and `RegressionModelParams` as frozen abstract base dataclasses. Single-task params classes (`LogisticRegressionParams`, `LassoParams`, `LinearRegressionParams`, `RidgeParams`, `ElasticNetParams`) now inherit from the appropriate base and no longer redeclare `task_type` or `scoring` fields.
* **Typed DecisionTree/RandomForest Params Subclasses**: Split `DecisionTreeParams` and `RandomForestParams` into abstract base classes and task-specific concrete subclasses (`DecisionTreeClassifierParams`, `DecisionTreeRegressorParams`, `RandomForestClassifierParams`, `RandomForestRegressorParams`). Each subclass encodes its own `task_type`, `scoring` default, criterion default, and `__post_init__` validation, eliminating the need for runtime `task_type` guards in model `__init__` methods.
* **`model_type` Property Refactor**: Removed `_model_type` backing instance variables from all nine concrete model classes; `model_type` properties now return enum constants directly.
* **Redundant `task_type` Property Removal**: Removed `_task_type` instance variables and override properties from the five single-task model classes; `task_type` is now inherited from `ClassificationModelSpecification` or `RegressionModelSpecification`.
* **Factory Task Routing Simplification**: Removed explicit `task_type` from `ModelSpecification.instantiate_model` and normalized base `DecisionTree` / `RandomForest` classes to task-specific wrappers during config assembly.
* **Model Inheritance Alignment**: Updated single-task model classes (Logistic, Linear, Lasso, Ridge, Elastic Net) to inherit from task-specialized base specifications.
* **Legacy Base Class Removal**: Removed `DecisionTree` and `RandomForest` dual-task base classes. `DecisionTreeClassifierModel`, `DecisionTreeRegressorModel`, `RandomForestClassifierModel`, and `RandomForestRegressorModel` now inherit directly from `ClassificationModelSpecification` or `RegressionModelSpecification`.
* **Memory Telemetry Units**: Standardized fit-time memory telemetry to GB values in both implementation and docstrings.
* **Audit Timestamp Precision**: Extended auto-generated `audit_timestamp` format from minute precision to second precision (`%Y%m%d_%H%M%S`) to prevent run/export collisions during rapid consecutive experiments.
* **`ModelAuditorConfig` Override API**: Added `overridable_fields()` classmethod and `apply_overrides(overrides)` instance method to `ModelAuditorConfig`. `apply_overrides` validates keys against the safe-to-mutate field set and applies them via `setattr`, returning the list of applied keys. `data_splits` is write-protected via `__setattr__` after initialization to guard core split consistency.
* **Fit-Time `verbose` Forwarding for Supported Estimators**: `ModelSpecification.fit()` now inspects estimator `fit(...)` signatures and forwards `verbose=self.verbose` only when supported, while continuing to pass `sample_weight` only when accepted. Model-level `verbose` defaults now source from `prefs.fit_verbose` (rather than `prefs.cv_verbose`).
* **Lazy Snapshot Reload Hydration**: `ModelAuditSummary.__setstate__` now restores snapshot state without eagerly recomputing missing validation/test prediction artifacts. Backfilling has been moved into explicit hydration helpers so reload-heavy workflows like export regeneration stay fast by default while still supporting on-demand artifact reconstruction when needed.
* **Configurable PDF Feature-Importance Cap**: Added `ModelAuditorConfig.pdf_feature_importance_chart_limit` and matching `ModelAuditSummary` support so the deep-dive PDF feature-importance subplot can be capped independently of the full Top N used elsewhere.
* **Anomaly Column Selection Priority**: When anomaly-table capping is enabled, dynamic context columns are now selected by model feature-importance rank when available, with deterministic fallback to original anomaly-feature order when importance data is unavailable.
* **Anomaly Header Disambiguation**: Data Anomaly Log dynamic feature headers now append OHE suffixes (for example, `DO Loc [071]`) so encoded variants are explicitly identifiable instead of being shown as generic duplicate labels.
* **Anomaly Note Visibility Toggle**: Added `ModelAuditorConfig.anomaly_table_show_notes` (default `True`) so presentation-focused reports can suppress Data Anomaly Log note text while preserving table rendering.

### Fixed

* **OHE Column Expansion on Snapshot Restore**: `ModelAuditSummary._init_features_to_fit_set()` now replicates the OHE-expansion logic from `ModelAuditor._resolve_features_to_fit_set()`. After loading a saved snapshot, the feature set is correctly resolved to one-hot-encoded column names (e.g. `DOLocationID_71`) rather than the original categorical column names, preventing `KeyError` failures when the test step tries to select features from the encoded DataFrame.
* **Non-ASCII Characters in PDF Chart Text**: Replaced Unicode glyphs (`\u03c3`, `\u0394`, `\u2026`, `\u2022`, and the warning sign `\u26a0`) used in rendered PDF labels, legend entries, and annotation text with ASCII-safe equivalents (`Std Dev`, `Delta`, `...`, `-`). This eliminates `UserWarning: Glyph ... missing from font(s) Arial` warnings emitted by Matplotlib when saving figures with the default Arial font.
* **Anomaly Page Readability Annotation**: Added a subtitle note on the Data Anomaly Log page when dynamic context columns are capped (for example, "Showing 8 of 24 anomaly context columns by feature importance for readability"), making truncation explicit to report readers.
* **Anomaly Compression Advisory Expansion**: Extended the Data Anomaly Log advisory logic to trigger whenever table columns are materially compressed for fit, including when `anomaly_table_max_columns` is already configured. Advisory text now recommends reducing the configured cap when compression persists (for example, when `Actual`/`Predicted` base columns become crowded).
* **Configurable Anomaly Note Suppression**: Data Anomaly Log cap and compression notes now respect `anomaly_table_show_notes`; when disabled, both note lines are omitted.

* **CloudPath Save Compatibility**: Updated `ModelAuditor.run_audit` and `ModelAuditSummary.export_results` to accept `CloudPath` destinations for audit logs and snapshot exports, using protocol-aware path handling instead of forced local `Path(...)` coercion.
* **String Target Scorer Compatibility**: Classification CV tuning now maps to weighted scorers (`f1_weighted`, `precision_weighted`, `recall_weighted`) to avoid `pos_label` errors with non-numeric labels.
* **Classification Anomaly Logic**: Replaced residual subtraction on string labels with misclassification-based anomaly flags.
* **Categorical Statistics Stability**: Added categorical target factorization in `ModelConfigurationStats` to prevent failures in mean/std/skew/kurtosis calculations.
* **PDF Categorical Target Distribution Rendering**: Updated `AuditPDFRenderer` target-distribution plotting to handle string/categorical targets with class-count bars instead of numeric percentile/KDE logic, preventing PDF export failures in classification audits.
* **Empty Inverse Transform Guard**: `DataSplits.inverse_transform_df` now short-circuits on empty DataFrames to prevent scaler shape errors.
* **Feature Metadata Initialization**: `ModelAuditorConfig.from_dataset` now auto-builds `FeatureMetadata` when not explicitly supplied, preventing empty fit-feature configurations.
* **Snapshot Score Stability**: Updated `ModelAuditSummary` state rehydration to preserve persisted validation/test scalar scores while backfilling missing prediction arrays.
* **Detailed Audit Stats RAM Display**: Corrected "Actual Peak RAM" formatting in the PDF detailed stats table to avoid double-scaling values already stored in GB.
* **Deep-Dive PDF Layout Consistency**: Stabilized deep-dive quadrant geometry and title placement across model types, including centered confusion matrix rendering, consistent right-margin bounds, and improved spacing for upper/lower quadrant titles and content.
* **Classification Misses Rendering**: Implemented robust lower-right "Top Validation Misses" table rendering for classification deep dives (including probability-availability fallbacks).
* **Regression Deep-Dive Balance**: Reduced lower-left residual chart height and aligned its vertical footprint with the lower-right misses table region for consistent page balance.
* **Regenerate Reload Cost**: Snapshot reloads no longer trigger eager `fit_and_evaluate_val()` / `evaluate_test_set_performance()` calls during unpickle, avoiding unnecessary recomputation when re-exporting existing audit artifacts.

## [1.2.3] - 2026-04-11

### Fixed

* **Link Optimization**: Replaced relative documentation links with absolute GitHub URLs to ensure cross-platform compatibility between GitHub and PyPI.

## [1.2.2] - 2026-04-11

### Fixed

* **README links**: Fixed links in README file for Sample Audit PDF and Sample Output.

## [1.2.1] - 2026-04-11

### Added

* **Sample Audit Artifacts**: Localized a full suite of sample outputs in `docs/artifacts/`, including the **Yellow Taxi Audit PDF**, **Excel Workbook**, and **JSON Snapshot**.
* **Interactive Audit State**: Provided a 118MB **Joblib** binary as a GitHub Release Asset, allowing users to hydrate and explore the complete `ModelAuditSummary` state from the October 2025 Taxi project.
* **Audit Documentation Badges**: Updated the README with dynamic status badges and direct links to generated reports for improved feature discoverability.

### Fixed

* **PyPI Metadata Alignment**: Synchronized the PyPI project description with the latest GitHub README to ensure the **Core Capabilities** and **Export Results** sections correctly reflect version 1.2.0's orchestration features.

## [1.2.0] - 2026-04-11

### Added

* **ModelAuditor Orchestrator**: introduced a centralized audit manager to handle model lifecycles, including tuning, final fitting, and automated report generation.
* **Positional Feature Alignment**: Added a deterministic sorting mechanism in `ModelFeatureImportance` using the `position` attribute to ensure feature names correctly map to importance arrays in unordered sets.
* **Audit Trace Logging**: Implemented `AuditLogger` to provide a "black box" recording of all terminal outputs (warnings, memory alerts, and fit status) into persistent log files.
* **Automated Model Viability Pruning**: Added logic to the orchestrator to automatically remove underperforming models from the competitive sweep based on a configurable `viable_score_gap`.

### Changed

* **Exact Upsampling/Downsampling**: Refactored `DataSplits` resampling strategies to use explicit row counts (`n=len(feat_min)`) instead of integer factors, ensuring perfectly balanced classes where `counts[0] == counts[1]`.
* **Memory-Safe Serialization**: Optimized `ModelAuditSummary` with a "side-car" extraction strategy for large prediction arrays during `JOBLIB` exports to prevent system OOM errors.
* **Enhanced Quality Scoring**: Updated `ModelConfigurationStats` to penalize the `quality_score` if models show extreme sensitivity to outliers (calculated via the gap between cleaned and raw validation scores).

### Fixed

* **Empty DataFrame Errors**: Corrected a failure where the `ModelAuditor` would attempt to fit models with zero columns by ensuring `FeatureMetadata` is properly propagated through the `ModelAuditorConfig`.
* **PDF Rendering Stability**: Resolved a `ValueError` in the `AuditPDFRenderer` by enforcing strict index and length alignment between validation targets and model predictions during residual analysis plotting.
* **Frozen Instance Errors**: Fixed immutability issues in `ModelConfiguration` tests by utilizing `dataclasses.replace` for injecting test-specific feature importance data.

### Technical Notes

* **Hardware Context**: Successfully verified peak RAM tracking and multi-core job propagation (`n_jobs`) across all model specifications.
* **Regression Integrity**: Confirmed that linear models correctly utilize absolute coefficients (`coef_`) for relative importance reporting in the deep-dive audit pages.

## [1.1.0] - 2026-02-10

### Documentation

* Documented new defaults for `FeatureMetadata.from_df` parameters `formatters` and `format_exceptions`.

## [1.0.0] - 2026-02-08

### Breaking

* Version reset to 1.0.0 to reflect non-backward-compatible changes across the library.

### Documentation

* Expanded module, class, and method docstrings across evaluation, models, and utilities.
