import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "/Users/scottroberts/Documents/Developer/Projects/BetaBank/datasets/Churn.csv"
OUT_PATH = "/Users/scottroberts/Documents/Developer/Projects/Python Libraries/dsr-feature-eng-ml/src/dsr_feature_eng_ml/reports/decision_tree_validate_model_report_real.txt"
HEADER = "Decision Tree (Auto-Balanced - Validation)"
REPORT_WIDTH = 80
SCORE_FORMAT = ".4f"

# Load data
df = pd.read_csv(DATA_PATH).head(500)
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
X = df[features]
y = df["Exited"]

# Split: main/test, then train/valid as in the library
X_main, X_test, y_main, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_main, y_main, test_size=0.25, random_state=42, stratify=y_main)

# Scale features using training stats
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_valid_s = scaler.transform(X_valid)

# Grid search for best params
param_grid = {
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
}
base = DecisionTreeClassifier(class_weight="balanced", random_state=42)
search = GridSearchCV(base, param_grid=param_grid, cv=3, scoring="f1")
search.fit(X_train_s, y_train)

best_params = search.best_params_
best_score = search.best_score_

# Train validation model with best params
val_model = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced",
    **best_params
)
val_model.fit(X_train_s, y_train)

# Predictions on validation set
pred_valid = pd.Series(val_model.predict(X_valid_s), index=y_valid.index)
frequency = pred_valid.value_counts(normalize=True)

# Build formatted report text
target_section = f"\n\nTarget Frequency: {frequency}\n"
header_line = HEADER.center(REPORT_WIDTH, "-")
params_line = f"Parameters: {best_params}"
cv_line = f"CV score:             {best_score:{SCORE_FORMAT}}"
train_line = f"Training Set score:   None"
valid_line = f"Validation Set score: None"
# Generalization is undefined without train/valid scores; mirror library default
generalization_line = f"Model Generalization: Undefined"

report_text = "\n".join([
    target_section.strip("\n"),
    header_line,
    params_line,
    cv_line,
    train_line,
    valid_line,
    generalization_line,
]) + "\n"

# Write output
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"Wrote report to: {OUT_PATH}")
