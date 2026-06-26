# Insurance Claim Amount Prediction

## Task Objective
Estimate medical insurance claim amounts (`charges`) based on personal data — age, sex, BMI,
number of children, smoking status, and region — and understand which factors drive cost the most.

## Approach
- **Cleaned the data:** checked for missing values (none) and duplicates (1 found, dropped)
- **Explored relationships:** visualized how age, BMI, and smoking status individually and
  jointly affect charges (scatter plots, boxplot, correlation heatmap)
- **Encoded categorical features:** One-Hot Encoded `sex`, `smoker`, `region` via a
  `ColumnTransformer` pipeline
- **Trained a model:** Linear Regression on an 80/20 train-test split
- **Evaluated performance:** MAE, RMSE, R², plus 5-fold cross-validation for stability
- **Interpreted the model:** ranked coefficients by effect size; checked predicted-vs-actual and
  residual plots for patterns the model misses

## Results and Insights

| Metric | Value |
|---|---|
| MAE | $4,177 |
| RMSE | $5,956 |
| R² | 0.81 |
| 5-fold CV MAE | $4,200 (± $89) |

- **Smoking status is the dominant driver of charges** — it adds roughly **$23,000** to predicted
  charges, far more than age (~$248/year) or BMI (~$319/unit).
- **Age** has a steady, fairly linear relationship with charges, regardless of smoking status.
- **BMI mostly matters in combination with smoking** — high-BMI smokers (BMI > ~30) form a
  visibly separate, much higher-cost cluster. A plain linear model can't fully represent this
  interaction, and it shows up as the main pattern in the residual plot.
- **Next step if extended:** add an explicit `bmi × smoker` interaction term, or try a
  non-linear model (Random Forest / Gradient Boosting) to close the remaining error.

---

**Files:** `Insurance_Claim_Prediction.ipynb` (notebook) · `insurance.csv` (Medical Cost Personal
Dataset, 1,338 records, 7 columns)

**Run it:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Insurance_Claim_Prediction.ipynb
```
Keep `insurance.csv` in the same folder as the notebook.
