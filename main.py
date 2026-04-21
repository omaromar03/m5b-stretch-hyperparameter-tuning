import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


# ======================
# 1) Setup
# ======================
os.makedirs("results", exist_ok=True)

DATA_PATH = "data/telecom_churn.csv"

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "has_partner",
    "has_dependents",
    "contract_months",
]


# ======================
# 2) Load data
# ======================
df = pd.read_csv(DATA_PATH)

X = df[NUMERIC_FEATURES]
y = df["churned"]


# ======================
# 3) Part 1 - GridSearchCV for Random Forest
# ======================
rf_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
}

inner_cv_part1 = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_grid = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    scoring="f1",
    cv=inner_cv_part1,
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X, y)

print("\n===== PART 1: GRID SEARCH RESULTS =====")
print("Best Params:", rf_grid.best_params_)
print("Best F1 Score:", rf_grid.best_score_)

# Save all grid results
grid_results = pd.DataFrame(rf_grid.cv_results_)
grid_results.to_csv("results/rf_gridsearch_results.csv", index=False)

# Heatmap:
# Fix min_samples_split at the best value, then show max_depth x n_estimators
best_split = rf_grid.best_params_["min_samples_split"]
filtered_results = grid_results[
    grid_results["param_min_samples_split"] == best_split
]

heatmap_data = filtered_results.pivot_table(
    index="param_max_depth",
    columns="param_n_estimators",
    values="mean_test_score"
)

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".4f")
plt.title(f"Mean F1 Score (min_samples_split={best_split})")
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.tight_layout()
plt.savefig("results/rf_heatmap.png")
plt.close()


# ======================
# 4) Helper function for Part 2 - Nested CV
# ======================
def run_nested_cv(model, param_grid, X, y, model_name):
    """
    Run nested cross-validation for one model family.

    Outer CV:
        Honest evaluation of the full tuning procedure
    Inner CV:
        GridSearchCV hyperparameter tuning

    Returns:
        summary_dict: summary metrics for final comparison table
        fold_results_df: detailed results for each outer fold
    """

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=123  # different from inner CV
    )

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    inner_best_scores = []
    outer_test_scores = []
    best_params_per_fold = []

    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train_outer = X.iloc[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_train_outer = y.iloc[train_idx]
        y_test_outer = y.iloc[test_idx]

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=inner_cv,
            n_jobs=-1,
            verbose=0
        )

        # Inner loop tuning
        grid.fit(X_train_outer, y_train_outer)

        # Best inner CV score from tuning
        inner_best_score = grid.best_score_
        inner_best_scores.append(inner_best_score)

        # Evaluate best tuned model on outer test fold
        best_model = grid.best_estimator_
        y_pred_outer = best_model.predict(X_test_outer)
        outer_test_f1 = f1_score(y_test_outer, y_pred_outer)

        outer_test_scores.append(outer_test_f1)
        best_params_per_fold.append(grid.best_params_)

        print(
            f"{model_name} | Fold {fold_num} | "
            f"Inner best F1 = {inner_best_score:.4f} | "
            f"Outer test F1 = {outer_test_f1:.4f} | "
            f"Best params = {grid.best_params_}"
        )

    fold_results_df = pd.DataFrame({
        "model": [model_name] * 5,
        "fold": [1, 2, 3, 4, 5],
        "inner_best_score": inner_best_scores,
        "outer_test_f1": outer_test_scores,
        "gap": np.array(inner_best_scores) - np.array(outer_test_scores),
        "best_params": best_params_per_fold,
    })

    summary_dict = {
        "Model": model_name,
        "Inner best_score_ (mean)": float(np.mean(inner_best_scores)),
        "Outer nested CV score (mean)": float(np.mean(outer_test_scores)),
        "Gap (inner - outer)": float(np.mean(inner_best_scores) - np.mean(outer_test_scores)),
    }

    return summary_dict, fold_results_df


# ======================
# 5) Part 2 - Nested CV
# ======================
print("\n===== PART 2: NESTED CV RESULTS =====")

# Random Forest family
rf_nested_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

rf_nested_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
}

# Decision Tree family
dt_nested_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

dt_nested_grid = {
    "max_depth": [3, 5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
}

rf_summary, rf_fold_results = run_nested_cv(
    model=rf_nested_model,
    param_grid=rf_nested_grid,
    X=X,
    y=y,
    model_name="Random Forest"
)

dt_summary, dt_fold_results = run_nested_cv(
    model=dt_nested_model,
    param_grid=dt_nested_grid,
    X=X,
    y=y,
    model_name="Decision Tree"
)

comparison_table = pd.DataFrame([rf_summary, dt_summary])

print("\n===== NESTED CV COMPARISON TABLE =====")
print(comparison_table)

# Save outputs
comparison_table.to_csv("results/nested_cv_comparison.csv", index=False)
rf_fold_results.to_csv("results/rf_nested_cv_folds.csv", index=False)
dt_fold_results.to_csv("results/dt_nested_cv_folds.csv", index=False)