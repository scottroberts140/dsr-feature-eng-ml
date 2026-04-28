# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* **Task-Specialized ModelSpecification Bases**: Introduced `ClassificationModelSpecification` and `RegressionModelSpecification` to centralize task-specific scoring behavior and reduce branching in shared model orchestration logic.
* **Automatic Categorical Encoding Trace**: Added audit log visibility when `DataSplits.from_data_source` auto-applies one-hot encoding, including the affected source columns.

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

### Fixed

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
