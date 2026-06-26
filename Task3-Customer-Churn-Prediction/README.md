# Customer Churn Prediction (Bank Customers)

## Task Objective
Identify which bank customers are likely to leave (churn), and understand what factors drive
that decision.

## Approach
- **Cleaned the data:** dropped non-predictive identifier columns (`RowNumber`, `CustomerId`,
  `Surname`); checked for missing values and duplicates (none found); noted the ~20% churn class
  imbalance
- **Explored relationships:** churn rate by geography, gender, age distribution, and number of
  products held
- **Encoded categorical features:** `Gender` via Label Encoding (binary), `Geography` via
  One-Hot Encoding (3 unordered categories)
- **Trained models:** Logistic Regression (interpretable baseline) and Random Forest (main
  model), both with `class_weight="balanced"` to handle the imbalance
- **Evaluated performance:** accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC
  curves, 5-fold cross-validation
- **Analyzed feature importance:** Random Forest importances compared against Logistic
  Regression coefficients

## Results and Insights

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.71 | 0.78 |
| **Random Forest** | **0.82** | **0.87** |

- **Random Forest clearly outperforms Logistic Regression** on every metric, confirmed stable
  via cross-validation.
- **Age is the strongest churn driver**, followed by Number of Products, Balance, Active Member
  status, and Geography (Germany).
- **Customers holding 3-4 products churn at 83-100%**, despite 2-product customers having the
  *lowest* churn rate (7.6%) — a non-monotonic pattern that Logistic Regression's linear
  coefficients miss entirely, but Random Forest captures.
- **Germany churns at roughly 2x the rate** of France or Spain.
- **Next step if extended:** investigate why 3+ product customers churn so heavily; try gradient
  boosting (XGBoost/LightGBM) for a possible AUC improvement.

---

**Files:** `Customer_Churn_Prediction.ipynb` (notebook) · `churn.csv` (Churn Modelling Dataset,
10,000 bank customers, 14 columns)

**Run it:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Customer_Churn_Prediction.ipynb
```
Keep `churn.csv` in the same folder as the notebook.
