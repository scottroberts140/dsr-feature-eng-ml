from dsr_feature_eng_ml.enums import BalancingStrategy, ModelEvaluationMethod
from dsr_feature_eng_ml.models.decision_tree import DecisionTree
from dsr_feature_eng_ml.evaluation.data_splits import DataSplits
import os
import sys
import pandas as pd

# Ensure the library source is on path
SRC_PATH = "/Users/scottroberts/Library/CloudStorage/GoogleDrive-scottrdeveloper@gmail.com/My Drive/Projects/Python Libraries/dsr-feature-eng-ml/src"
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


# Paths
DATA_PATH = (
    "/Users/scottroberts/Library/CloudStorage/GoogleDrive-scottrdeveloper@gmail.com/My Drive/Projects/BetaBank/datasets/Churn.csv"
)
OUT_PATH = "/Users/scottroberts/Library/CloudStorage/GoogleDrive-scottrdeveloper@gmail.com/My Drive/Projects/Python Libraries/dsr-feature-eng-ml/src/dsr_feature_eng_ml/reports/decision_tree_validate_model_report_real.txt"

# Load a small subset of the dataset
df = pd.read_csv(DATA_PATH)
# Use a modest subset to keep it fast
df = df.head(500)

# Feature/target selection (numeric-only to avoid encoding)
features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

# Create data splits
splits = DataSplits.from_data_source(
    src=df,
    features_to_include=features,
    target_column="Exited",
    test_size=0.2,
    valid_size=0.25,
    random_state=42,
    scale_features=True,
    shuffle=True,
    stratify=True,
)

# Minimal param grid for speed
param_grid = {
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
}

# Instantiate DecisionTree
model = DecisionTree(
    data_splits=splits,
    cv=3,
    param_grid=param_grid,
    class_weight="balanced",
    n_iter=10,
)

# Tune then validate
model.tune_hyperparameters()
mc, report_text = model.validate_model(
    model_balancing=BalancingStrategy.WEIGHTED,
    evaluation_method=ModelEvaluationMethod.VALIDATION_SET,
    header="Decision Tree (Auto-Balanced - Validation)",
    include_in_report=True,
    show_plot=False,
)

# Write the report text to file
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"Wrote report to: {OUT_PATH}")
