# Customer Churn Prediction (Bank Customers)

Predicts which bank customers are likely to churn, and identifies the key drivers behind it.

## Dataset
- **File:** `churn.csv`
- **Source:** Churn Modelling Dataset
- **Size:** 10,000 bank customers, 14 columns (target: `Exited`)

## Notebook
`Customer_Churn_Prediction.ipynb`

## What it does
- Cleans the data (drops ID columns, checks nulls/duplicates)
- Encodes `Gender` (Label Encoding) and `Geography` (One-Hot Encoding)
- Trains Logistic Regression and Random Forest classifiers
- Evaluates with accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- Analyzes feature importance to explain what drives churn

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.71 | 0.78 |
| **Random Forest** | **0.82** | **0.87** |

**Top churn drivers:** Age, Number of Products, Balance, Active Member status, Geography (Germany).

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest)
- Matplotlib, Seaborn

## Project Structure
```
.
├── README.md
├── Customer_Churn_Prediction.ipynb
└── churn.csv
```

## Run it
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Customer_Churn_Prediction.ipynb
```
Keep `churn.csv` in the same folder as the notebook.

