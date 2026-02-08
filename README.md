# dsr-feature-eng-ml

Comprehensive machine learning model evaluation and feature engineering framework.

## Features

- **Model Evaluation**: Automatic hyperparameter tuning and model comparison for Decision Trees, Random Forests, and Logistic Regression
- **Data Balancing**: Support for imbalanced dataset handling (upsampling, downsampling, balanced class weights)
- **Feature Importance**: Automatic feature selection and importance ranking
- **Data Splitting**: Intelligent train/validation/test splitting with automatic feature scaling
- **Result Tracking**: Comprehensive model configuration and performance metrics tracking

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
- Fits scaler on training data only (prevents data leakage)
- Transforms validation and test sets consistently
- Supports upsampling and downsampling for class imbalance

### ModelEvaluation
Orchestrates comprehensive model evaluation:
- Evaluates multiple model types in parallel
- Supports four balancing strategies
- Tracks best performing models
- Generates detailed evaluation reports

### Model Classes
- **DecisionTree**: Decision Tree classifier with feature importance
- **RandomForest**: Random Forest classifier with ensemble methods
- **LogisticRegression**: Logistic Regression with convergence control

## Requirements

- Python >= 3.9
- pandas
- numpy
- scikit-learn >= 1.0
- dsr-data-tools
- dsr-utils

## Architecture

The library uses a modular approach:
- `evaluation/`: Core evaluation pipeline (DataSplits, ModelEvaluation, ModelResults)
- `models/`: Model implementations and hyperparameter tuning
- `enums.py`: Enumeration types for model states and configurations
- `constants.py`: Global configuration and defaults

## Preferences and Overrides

You can override library defaults (like constants used in evaluation and reporting) without changing code in the library.

**Precedence (highest to lowest)**
- Runtime override via `set_pref()`
- Environment variables prefixed with `DSR_FEML_`
- User config file in `~/.config/dsr-feature-eng-ml/config.toml` or `~/Library/Application Support/dsr-feature-eng-ml/config.toml`
- Project-level `./dsr_feature_eng_ml.toml`
- In-library default value

**Examples**
- Runtime (Python):
    ```python
    from dsr_feature_eng_ml import set_pref
    set_pref("REPORT_WIDTH", 120)
    set_pref("SCORE_FORMAT", ".3f")
    ```
- Environment (shell):
    ```bash
    export DSR_FEML_REPORT_WIDTH=120
    export DSR_FEML_SCORE_FORMAT=.3f
    export DSR_FEML_DEFAULT_ACCEPTABLE_GAP=0.03
    ```
- Config file (TOML):
    ```toml
    [constants]
    REPORT_WIDTH = 120
    SCORE_FORMAT = ".3f"
    DEFAULT_ACCEPTABLE_GAP = 0.03
    ```

**How it works**
- `constants.py` defines defaults and resolves effective values through the preferences system:
    ```python
    from dsr_feature_eng_ml.preferences import resolve_constant
    SCORE_FORMAT = resolve_constant("SCORE_FORMAT", ".4f")
    REPORT_WIDTH = resolve_constant("REPORT_WIDTH", 100)
    ```
- Most code should continue to import these constants (e.g., `from dsr_feature_eng_ml import REPORT_WIDTH`).

**Should I call `resolve_constant()` directly?**
- No for typical usage: import constants as usual, they already reflect preferences at import time.
- Yes if you need late-binding (e.g., react to `set_pref()` after modules are imported). In that case, call `get_pref("REPORT_WIDTH", 100)` or `resolve_constant("REPORT_WIDTH", 100)` where you need the value.

This keeps defaults centralized while giving users clean override hooks at runtime, via environment, or via config files.

## License

MIT License - see LICENSE file for details
