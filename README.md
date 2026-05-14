# dsr-feature-eng-ml

[![PyPI version](https://img.shields.io/pypi/v/dsr-feature-eng-ml.svg?cacheSeconds=300)](https://pypi.org/project/dsr-feature-eng-ml/)
[![Python versions](https://img.shields.io/pypi/pyversions/dsr-feature-eng-ml.svg?cacheSeconds=300)](https://pypi.org/project/dsr-feature-eng-ml/)
[![License](https://img.shields.io/pypi/l/dsr-feature-eng-ml.svg?cacheSeconds=300)](https://pypi.org/project/dsr-feature-eng-ml/)
[![Changelog](https://img.shields.io/badge/changelog-available-blue.svg)](https://github.com/scottroberts140/dsr-feature-eng-ml/releases)
[![Sample Audit PDF](https://img.shields.io/badge/Audit-Sample_Report-blue.svg)](https://github.com/scottroberts140/dsr-feature-eng-ml/blob/main/docs/artifacts/Taxi%20-%20202510_20260208_1600.pdf)

This suite provides a high-fidelity framework for training, evaluating, and auditing machine learning models. It is designed to move beyond simple accuracy metrics, providing deep insights into model generalization, data drift, and hardware efficiency.

**Version 1.3.4**: This release adds `min_child_weight` support for XGBoost classifier parameters, including estimator wiring, parameter reporting, and standard search-grid coverage.

**Version 1.3.1**: This release improves deep-dive PDF reliability by fixing hyperparameter table border/fit behavior and auto-hydrating missing validation prediction artifacts when rendering from restored snapshots.

**Release scope**: Regression workflows have been tested. Classification workflows include native ROC-AUC metrics and comprehensive test coverage (66 tests total across dsr-orchestrator and dsr-feature-eng-ml).

## Version 1.3.0 Implementation Details

* Added task-specialized model bases: `ClassificationModelSpecification` and `RegressionModelSpecification`.
* Introduced `ClassificationModelParams` and `RegressionModelParams` abstract base dataclasses; single-task params classes now inherit task-appropriate defaults without redeclaring `task_type` or `scoring`.
* Split `DecisionTreeParams` and `RandomForestParams` into abstract bases with task-specific concrete subclasses (`DecisionTreeClassifierParams`, `DecisionTreeRegressorParams`, `RandomForestClassifierParams`, `RandomForestRegressorParams`), removing runtime task-type guards from model constructors.
* Simplified factory construction by removing explicit task routing from `instantiate_model` and normalizing legacy base classes to task-specific wrappers.
* Updated fit-time memory telemetry to GB units and aligned related docstrings.
* Added automatic one-hot encoding trace logging in `DataSplits.from_data_source` (including encoded source columns).
* Improved classification scoring robustness for string labels by using weighted CV scorers where required.
* Increased audit timestamp precision to seconds (`%Y%m%d_%H%M%S`) to avoid output/snapshot collisions when running multiple experiments within the same minute.
* Improved deep-dive PDF layout stability and consistency across classification/regression pages, including centered confusion matrices, consistent quadrant bounds, and refined title/content spacing.
* Added/strengthened classification deep-dive fallback rendering when probability/importance data is unavailable.
* Corrected Detailed Audit Stats "Actual Peak RAM" formatting so values already stored in GB render accurately.
* Added `ModelAuditorConfig.overridable_fields()` and `apply_overrides(overrides)` to support validated post-construction field patching. `data_splits` is write-protected after initialization to maintain split consistency. Invalid or unknown keys raise `ValueError`; private-field assignment raises `AttributeError`.
* Extended `AuditLogger` with Python `logging` integration: a `FileHandler` is now attached to the root logger on entry so that `logging.info()`, `logging.warning()`, and similar calls are captured in the same log file as stdout. The `log_level` parameter (default `INFO`) sets the minimum level written to the file. The tee behavior (stdout to both terminal and file) is unchanged. `ModelAuditor.run_audit` exposes this via an `audit_log_level` parameter.
* Added live stage-by-stage progress output during `run_audit`. Each model prints `Tuning` and `Fitting` headers with indented sub-stage messages covering data preparation, row sampling, memory checks, search execution (type, candidates/params, CV, workers, duration), fit start, training completion, and validation completion. Post-loop finalization messages and a per-model progress summary (`Progress: N/M models recorded`) are also printed. Both `tune_model` and `fit_and_evaluate_val` now accept an optional `progress_callback` parameter.
* Added a separate fit-time verbosity preference, `prefs.fit_verbose`, distinct from `prefs.cv_verbose`. `cv_verbose` continues to control `GridSearchCV`/`RandomizedSearchCV` output, while `fit_verbose` controls estimator fit-time verbosity for model types whose `fit(...)` methods accept a `verbose` parameter.
* Updated `ModelSpecification.fit()` to forward `verbose=self.verbose` and `sample_weight` only when each parameter is supported by the estimator `fit(...)` signature, preventing unsupported-kwarg errors while enabling fit-time verbosity where available.
* Changed `ModelAuditSummary` snapshot reload behavior to be lazy by default. Loading a saved `Audit_State` no longer eagerly refits models just to backfill missing predictions; those artifacts are now hydrated explicitly only when a downstream path actually requests them.
* Made the PDF deep-dive feature-importance chart cap configurable through `ModelAuditorConfig.pdf_feature_importance_chart_limit`, so dense OHE-heavy audits can keep readable PDF pages without truncating richer exports.
* Added `ModelAuditorConfig.anomaly_table_max_columns` to optionally cap dynamic context columns on the Data Anomaly Log page.
* When anomaly-table capping is active, selected columns are prioritized by model feature importance when available (with deterministic fallback to original anomaly-feature order), and the PDF displays a subtitle note indicating that only a capped subset is shown for readability.
* Updated Data Anomaly Log dynamic headers to disambiguate one-hot encoded columns by including encoded suffixes (for example, `DO Loc [071]`) instead of ambiguous duplicate counters.
* Expanded anomaly-table readability guidance: when columns are compressed for fit (including cases where key base columns such as `Actual`/`Predicted` become crowded), the PDF now emits an advisory note recommending setting or lowering `anomaly_table_max_columns`.
* Added `ModelAuditorConfig.anomaly_table_show_notes` (default `True`) to control Data Anomaly Log annotation visibility. Set it to `False` to suppress both anomaly note lines for cleaner presentation PDFs.

## Core Capabilities

* **Automated Model Auditing**: Orchestrates competitive sweeps across multiple model architectures with built-in hyperparameter tuning and cross-validation.
* **Statistical Drift Analysis**: Automatically calculates mean and standard deviation deltas, skewness, and kurtosis across train/val/test splits to identify data inconsistency.
* **Intelligent Resampling**: Features exact balancing strategies for classification tasks, ensuring minority and majority classes are perfectly aligned during training.
* **Memory-Safe Operations**: Includes predictive memory auditing to prevent Out-of-Memory (OOM) errors during large-scale tuning and a side-car serialization strategy for handling massive prediction arrays.
* **Comprehensive Reporting**: Generates multi-page PDF audit reports featuring leaderboards, deep-dive residual analysis, and feature importance visualizations. [[View Sample PDF Report]](https://github.com/scottroberts140/dsr-feature-eng-ml/blob/main/docs/artifacts/Taxi%20-%20202510_20260208_1600.pdf)
* **Class-Balance Visibility**: Includes a Data Profile PDF page that summarizes train/validation/test target distributions, per-class percentages, and imbalance ratios for classification workflows.

## Multi-Format Export Capabilities

The ModelAuditor provides a robust export engine via the ModelAuditSummary class, allowing audit results to be persisted in several formats for different downstream use cases:

* **Audit PDF Report**: Generates a high-fidelity visual document including the executive summary, model leaderboards, and deep-dive residual analysis.
* **JOBLIB Snapshot**: Persists the full, executable state of the ModelAuditSummary object. This includes a memory-optimized "side-car" process that detaches large prediction arrays during the write operation to ensure system stability before reattaching them. Snapshot reload is lazy by default, so existing summary exports can be regenerated without eagerly rehydrating missing prediction artifacts.
* **Excel Workbook**: Creates a multi-sheet report containing the Audit Summary (metadata), the Leaderboard (performance results), an Anomaly Log (outlier data), and comprehensive Feature Metadata.
* **JSON Payload**: Exports a serializable, nested dictionary containing the complete audit snapshot, metadata, and per-model results, suitable for web integration or programmatic review.
* **CSV Collection**: Produces a set of tabular files for flat-file analysis, including distinct files for the leaderboard results, metadata summary, anomaly data, and dynamic feature context.
[[Sample Output]](https://github.com/scottroberts140/dsr-feature-eng-ml/tree/main/docs/artifacts)

## Audit Metrics Definitions

* **Quality Score**: A 0–100 metric assessing model stability; penalized if "cleaned" performance (after outlier removal) significantly diverges from raw performance.
* **Drift Index**: The percentage difference between training and validation target means, used to identify potential data shift.
* **Generalization Gap**: The absolute difference between training and validation scores (e.g., R² Gap); used to classify models as Well-Fit, Marginal, or Overfit.
* **Efficiency**: Measured in rows processed per second, providing context on model throughput relative to hardware resources.
* **ROC-AUC (Classification)**: The area under the receiver operating characteristic curve (0.0–1.0), measuring classification performance across probability thresholds. Binary classification uses `probs[:, 1]` for positive class probability; multiclass uses one-vs-rest (OvR) with weighted averaging. ROC-AUC values are computed for train, validation, cleaned validation, and test splits and are automatically included in model comparison leaderboards.

## Installation

```bash
pip install dsr-feature-eng-ml
```

## Quick Start

```python
import pandas as pd
from dsr_feature_eng_ml import DataSplits, ModelEvaluation

# Load your data
df = pd.read_csv('data.csv')

# Create data splits (with automatic scaling)
data_splits = DataSplits.from_data_source(
    src=df,
    features_to_include=['feature1', 'feature2', 'feature3'],
    target_column='target',
    test_size=0.2,
    valid_size=0.25,
    random_state=42,
    scale_features=True
)

# Evaluate models
results = ModelEvaluation.evaluate_dataset(
    data_splits=data_splits,
    dtree_param_grid={'max_depth': [5, 10, 20]},
    rf_param_grid={'n_estimators': [50, 100]},
    lr_param_grid={'C': [0.1, 1.0, 10.0]},
    cv=5,
    n_iter=50,
    max_iter=1000,
    scoring='f1',
    n_jobs=-1,
    viable_f1_gap=0.01,
    report_title='Model Evaluation',
    perform_dtree_feature_selection=True,
    perform_rf_feature_selection=True
)
```

## Key Components

### DataSplits

Manages train/validation/test splits with automatic feature scaling:

* Fits scaler on training data only (prevents data leakage)
* Transforms validation and test sets consistently
* Supports upsampling and downsampling for class imbalance

### ModelEvaluation

Orchestrates comprehensive model evaluation:

* Evaluates multiple model types in parallel
* Supports four balancing strategies
* Tracks best performing models
* Generates detailed evaluation reports

### Model Classes

* **DecisionTree**: Decision Tree classifier with feature importance
* **RandomForest**: Random Forest classifier with ensemble methods
* **LogisticRegression**: Logistic Regression with convergence control

## Requirements

* Python >= 3.11
* dsr-utils >= 1.7.3
* dsr-data-tools >= 2.1.0
* dsr-files >= 3.1.1
* numpy >= 2.4.4
* pandas >= 3.0.2
* scikit-learn >= 1.8.0
* matplotlib >= 3.10.8
* seaborn >= 0.13.2

## Architecture

The library uses a modular approach:

* `evaluation/`: Core evaluation pipeline (DataSplits, ModelEvaluation, ModelResults)
* `models/`: Model implementations and hyperparameter tuning
* `enums.py`: Enumeration types for model states and configurations
* `constants.py`: Global configuration and defaults

## Preferences and Overrides

You can override library defaults (like constants used in evaluation and reporting) without changing code in the library.

## Precedence (highest to lowest)

* Runtime override via `set_pref()`
* Environment variables prefixed with `DSR_FEML_`
* User config file in `~/.config/dsr-feature-eng-ml/config.toml` or `~/Library/Application Support/dsr-feature-eng-ml/config.toml`
* Project-level `./dsr_feature_eng_ml.toml`
* In-library default value

## Examples

* Runtime (Python):

    ```python
    from dsr_feature_eng_ml import set_pref
    set_pref("REPORT_WIDTH", 120)
    set_pref("SCORE_FORMAT", ".3f")
    ```

* Environment (shell):

    ```bash
    export DSR_FEML_REPORT_WIDTH=120
    export DSR_FEML_SCORE_FORMAT=.3f
    export DSR_FEML_DEFAULT_ACCEPTABLE_GAP=0.03
    ```

* Config file (TOML):

    ```toml
    [constants]
    REPORT_WIDTH = 120
    SCORE_FORMAT = ".3f"
    DEFAULT_ACCEPTABLE_GAP = 0.03
    ```

## How it works

* `constants.py` defines defaults and resolves effective values through the preferences system:

    ```python
    from dsr_feature_eng_ml.preferences import resolve_constant
    SCORE_FORMAT = resolve_constant("SCORE_FORMAT", ".4f")
    REPORT_WIDTH = resolve_constant("REPORT_WIDTH", 100)
    ```

* Most code should continue to import these constants (e.g., `from dsr_feature_eng_ml import REPORT_WIDTH`).

**Should I call `resolve_constant()` directly?**

* No for typical usage: import constants as usual, they already reflect preferences at import time.
* Yes if you need late-binding (e.g., react to `set_pref()` after modules are imported). In that case, call `get_pref("REPORT_WIDTH", 100)` or `resolve_constant("REPORT_WIDTH", 100)` where you need the value.

This keeps defaults centralized while giving users clean override hooks at runtime, via environment, or via config files.

## License

MIT License - see LICENSE file for details
