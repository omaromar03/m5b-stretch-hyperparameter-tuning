# m5b-stretch-hyperparameter-tuning

##  Overview

This project implements systematic hyperparameter tuning using GridSearchCV and evaluates model performance using nested cross-validation.

The goal is to:
- Identify optimal hyperparameters for a Random Forest model
- Analyze model complexity and performance trends
- Measure selection bias using nested cross-validation
- Compare Random Forest and Decision Tree performance

---

##  Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

Run the script:

python main.py
##  Dataset

Telecom churn dataset with numeric features:

senior_citizen
tenure
monthly_charges
total_charges
num_support_calls
has_partner
has_dependents
contract_months

Target variable:

churned (0 or 1)
---
##  Part 1 — GridSearchCV

We performed hyperparameter tuning on a Random Forest model using:

5-fold stratified cross-validation
Scoring metric: F1 score
Parameter Grid
n_estimators: [50, 100, 200]
max_depth: [3, 5, 10, 20, None]
min_samples_split: [2, 5, 10]
Results

Best parameters:

{'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 50}

Best F1 score:

0.4918
Heatmap

Saved at:

results/rf_heatmap.png
Analysis

The results show that max_depth has the largest impact on model performance. Shallow trees (depth 3–5) achieve the highest F1 scores, while deeper trees (depth 10 and 20) lead to a significant drop in performance, indicating strong overfitting.

There is a clear sweet spot around max_depth=5 and n_estimators=50, where the model achieves its best performance. Increasing the number of estimators beyond this point does not significantly improve performance, suggesting diminishing returns.

Overall, the model is more prone to overfitting than underfitting. Future tuning could focus on moderate depths and additional regularization parameters such as min_samples_leaf.

🔁 Part 2 — Nested Cross-Validation

Nested cross-validation was used to obtain an unbiased estimate of model performance and measure selection bias.

Method
Outer loop: 5-fold StratifiedKFold
Inner loop: GridSearchCV (same parameter grid)

For each outer fold:

Perform hyperparameter tuning on the training split
Evaluate the best model on the outer test split
📊 Results
Model	Inner Best Score	Outer Score	Gap
Random Forest	0.4967	0.4919	0.0048
Decision Tree	0.4765	0.4629	0.0135

Saved at:

results/nested_cv_comparison.csv
🧠 Analysis

The Decision Tree shows a larger gap between the inner best score and the outer nested CV score compared to the Random Forest. This indicates higher selection bias.

This behavior is expected because Decision Trees are high-variance models, meaning their optimal hyperparameters are more sensitive to the training data.

In contrast, Random Forest reduces variance through bagging, making it more stable and less prone to overfitting. As a result, its selection bias is smaller.

For Random Forest, the GridSearchCV.best_score_ is relatively reliable due to the small gap. However, for Decision Tree, the larger gap suggests that the inner CV score is more optimistic and less trustworthy.

This reflects the same principle as held-out test sets: data used for model selection should not be used for evaluation. Nested cross-validation ensures evaluation is performed on unseen data, providing a more honest estimate of performance.

## Outputs
results/rf_heatmap.png
results/rf_gridsearch_results.csv
results/nested_cv_comparison.csv
results/rf_nested_cv_folds.csv
results/dt_nested_cv_folds.csv
🚀 Conclusion
Random Forest outperforms Decision Tree on this dataset
Model performance is highly sensitive to tree depth
Overfitting is a key issue at higher depths
Nested cross-validation provides a more realistic estimate of performance
Selection bias is significantly higher for Decision Trees