# Fraud Detection — Financial Transaction Classifier

An end-to-end machine learning project that detects fraudulent financial transactions using Logistic Regression and serves predictions through an interactive Streamlit web app.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the App](#running-the-app)
- [Key Findings](#key-findings)

---

## Overview

The project covers three stages:

1. **Exploratory Data Analysis** — class distribution, fraud patterns by transaction type, amount distribution, and suspicious balance patterns.
2. **Feature Engineering + Model Training** — derived balance-difference features fed into a scikit-learn Pipeline (StandardScaler + OneHotEncoder + Logistic Regression).
3. **Deployment** — real-time prediction via a Streamlit web interface.

---

## Dataset

- **File:** `AIML Dataset.csv` *(not tracked — see .gitignore)*
- **Rows:** 6,362,620 transactions
- **Columns:** 11 (step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud)
- **Fraud rate:** 0.13% — severe class imbalance
- **Source:** Synthetic mobile money transaction log (PaySim-style)

---

## Features

| Feature | Type | Description |
|---|---|---|
| `step` | Numerical | Time unit (1 step = 1 hour) |
| `type` | Categorical | PAYMENT, TRANSFER, CASH\_OUT, DEBIT, CASH\_IN |
| `amount` | Numerical | Transaction amount |
| `oldbalanceOrg` | Numerical | Sender balance before |
| `newbalanceOrig` | Numerical | Sender balance after |
| `oldbalanceDest` | Numerical | Receiver balance before |
| `newbalanceDest` | Numerical | Receiver balance after |
| `balanceDiffOrig` | Engineered | `oldbalanceOrg - newbalanceOrig` |
| `balanceDiffDest` | Engineered | `newbalanceDest - oldbalanceDest` |

> `nameOrig`, `nameDest`, and `isFlaggedFraud` are excluded from the model.

---

## Model

- **Algorithm:** Logistic Regression
- **Pipeline:** `ColumnTransformer` (StandardScaler for numerics, OneHotEncoder for `type`) → `LogisticRegression`
- **Split:** Stratified 80/20 train-test split
- **Serialised as:** `fraud_detection_model.pkl`

---

## Project Structure

```
FraudDetection/
├── analysis_model.ipynb          # EDA and model training notebook
├── fraud_detection.py            # Streamlit prediction app
├── fraud_detection_model.pkl     # Trained pipeline (serialised)
├── fraud_detection_report.tex    # LaTeX project report
├── README.md                     # This file
└── AIML Dataset.csv              # Raw data (gitignored — 493 MB)
```

---

## Setup and Installation

**1. Clone the repository**
```bash
git clone https://github.com/dv919/Fraud_Detection.git
cd Fraud_Detection
```

**2. Create a virtual environment**
```bash
python -m venv .venv
```

**3. Activate it**
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

**4. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

**5. Add the dataset**

Download `AIML Dataset.csv` and place it in the project root (it is not tracked due to its size).

---

## Running the App

```bash
streamlit run fraud_detection.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Inputs required:**
- Transaction Type, Step, Amount
- Old/New Balance for both sender and receiver

`balanceDiffOrig` and `balanceDiffDest` are computed automatically.

---

## Key Findings

- Fraud occurs **only in TRANSFER and CASH\_OUT** transactions.
- A **zero-balance-after-transfer** pattern (1,188,074 cases) is a strong fraud signal.
- Engineered balance-difference features improve the linear model's ability to detect fraud.
- With only 0.13% fraud, **precision, recall, F1, and AUC-ROC** are more meaningful than accuracy.
