# Telecom Churn Prediction (CatBoost)

I built this project as a Junior ML Engineer to show an end-to-end ML workflow: data cleaning, feature engineering, modeling, and business-ready insights. The goal is to predict customer churn for a telecom provider and help retain users before they cancel.

## Goal and business value
- Predict whether a customer will terminate their contract (binary classification).
- Reduce churn by flagging at-risk clients early and enabling targeted retention actions.

## Data
Sources (merged by `customerID`, snapshot as of 2020-02-01):
- `contract_new.csv` - contract details, payments, dates, charges.
- `personal_new.csv` - demographics.
- `internet_new.csv` - internet services and add-ons.
- `phone_new.csv` - phone services.

## Approach
- Clean and consolidate multiple tables, handle dates and missing values.
- Create the target label: `IsBrokenContract`.
- Engineer features such as `ContractDuration` and `NumInternetServices`.
- Remove target leakage (e.g., `EndDate`).
- Build a reproducible pipeline with categorical encoding and numeric scaling.

## Models
I compare baseline and gradient-boosting models with hyperparameter tuning:
- Logistic Regression
- Decision Tree
- CatBoost
- LightGBM

Metric: ROC-AUC (good for ranking churn risk).

## Results
- **CatBoost (best model)**: ROC-AUC **0.929** on test, **0.916** during tuning.
- LightGBM: ROC-AUC **0.886**.
- Decision Tree: ROC-AUC **0.779**.
- Logistic Regression: ROC-AUC **0.755**.
- Dummy baselines: ~**0.50** (sanity check).

The model is strong on the "non-churn" class but misses some churners (FN). In practice, the decision threshold should be tuned to business costs (lost customer vs retention offer).

## Interpretation and business insights
Key risk drivers I found:
- Short customer tenure (the first months/year are critical).
- High `MonthlyCharges` (premium plans need a clearer value proposition).
- Contract type and payment habits.
- Service bundle and add-on saturation.

Recommendations:
- Strengthen onboarding and early-stage retention.
- For premium plans, use targeted "value bundles," not only discounts.
- Revisit long-term contract terms (clarity, benefits, flexibility).
- Use two threshold strategies: "low-cost retention" vs "high-cost retention."

## Stack
Python, pandas, numpy, scikit-learn, CatBoost, LightGBM, matplotlib, seaborn, phik, SHAP.

## How to run
1. Open `telecom_churn_catboost.ipynb` in Jupyter/VS Code.
2. Install dependencies (example):
   ```bash
   pip install -U pandas numpy scikit-learn catboost lightgbm matplotlib seaborn phik shap
   ```
3. Run the notebook top-to-bottom.

## Project structure
- `telecom_churn_catboost.ipynb` - main analysis, modeling, and conclusions.

