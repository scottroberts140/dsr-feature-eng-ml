# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
