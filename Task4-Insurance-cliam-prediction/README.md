# Insurance Claim Amount Prediction

Estimates medical insurance charges based on personal data, and shows what drives the cost.

## Dataset
- **File:** `insurance.csv`
- **Source:** Medical Cost Personal Dataset
- **Size:** 1,338 records, 7 columns (target: `charges`)

## Notebook
`Insurance_Claim_Prediction.ipynb`

## What it does
- Cleans the data (checks nulls/duplicates, drops 1 duplicate row)
- Visualizes how age, BMI, and smoking status affect charges
- One-Hot Encodes `sex`, `smoker`, `region`
- Trains a Linear Regression model
- Evaluates with MAE, RMSE, R², 5-fold cross-validation
- Interprets model coefficients to explain what drives charges

## Results

| Metric | Value |
|---|---|
| MAE | $4,177 |
| RMSE | $5,956 |
| R² | 0.81 |

**Top driver of charges:** Smoking status (~$23,000 added), far ahead of age and BMI.

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (Linear Regression)
- Matplotlib, Seaborn

## Project Structure
```
.
├── README.md
├── Insurance_Claim_Prediction.ipynb
└── insurance.csv
```

## Run it
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Insurance_Claim_Prediction.ipynb
```
Keep `insurance.csv` in the same folder as the notebook.

