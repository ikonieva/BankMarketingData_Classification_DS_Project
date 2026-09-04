# Bank Marketing Data — Term Deposit Classification

## The Task
The [Bank Marketing Data Set](https://archive.ics.uci.edu/dataset/222/bank+marketing) from the UCI Machine Learning Repository relates to direct marketing campaigns (phone calls) of a Portuguese banking institution.

The classification goal is to predict whether a client will subscribe to a term deposit (target variable `y`).

## Dataset

| Group | Feature | Description |
|---|---|---|
| **Client data** | `age` | Client's age (numeric) |
| | `job` | Type of job (categorical) |
| | `marital` | Marital status (categorical) |
| | `education` | Education level (categorical) |
| | `default` | Has credit in default? (categorical) |
| | `housing` | Has a housing loan? (categorical) |
| | `loan` | Has a personal loan? (categorical) |
| **Last contact of current campaign** | `contact` | Contact communication type (categorical) |
| | `month` | Last contact month (categorical) |
| | `day_of_week` | Last contact day of the week (categorical) |
| | `duration` | Last contact duration, in seconds (numeric) — ⚠️ **known data-leakage field**: `duration=0` implies `y=no`, but the value is unknown before a call is placed and is known only after the outcome (`y`) is already determined. Used only as a benchmark comparison, **excluded from the final predictive model**. |
| **Other attributes** | `campaign` | Number of contacts for this client during this campaign (numeric) |
| | `pdays` | Days since the client was last contacted in a previous campaign (numeric; `999` = never contacted) |
| | `previous` | Number of contacts before this campaign for this client (numeric) |
| | `poutcome` | Outcome of the previous marketing campaign (categorical: `failure`, `nonexistent`, `success`) |
| **Social/economic context** | `emp.var.rate` | Employment variation rate — quarterly (numeric) |
| | `cons.price.idx` | Consumer price index — monthly (numeric) |
| | `cons.conf.idx` | Consumer confidence index — monthly (numeric) |
| | `euribor3m` | Euribor 3-month rate — daily (numeric) |
| | `nr.employed` | Number of employees — quarterly (numeric) |
| **Target** | `y` | Has the client subscribed to a term deposit? (binary: `yes`/`no`) |

The dataset has no missing values, but the target class is strongly imbalanced and several numeric features contain outliers.

## Exploratory Data Analysis — Key Findings

- Single students and married admin staff are more likely to subscribe to a term deposit; blue-collar and services workers, and single/divorced technicians, are more likely to decline.
- Clients with a defaulted credit are almost never offered a term deposit; clients with a university degree are more likely to subscribe.
- Contact timing matters by job: admins contacted on Friday convert less; Wednesday/Thursday works better for them. Technicians convert best on Friday and worst on Thursday. No strong day-of-week effect for most other professions.
- Strongly correlated numeric features among clients who subscribed: `euribor3m` ↔ `emp.var.rate`, `nr.employed` ↔ `euribor3m`, `emp.var.rate` ↔ `nr.employed`, `cons.price.idx` ↔ `emp.var.rate`, and `pdays` ↔ `previous`.
- **Logical inconsistencies in contact history**: ~4,110 records (~10% of the data) have `pdays = 999` (never previously contacted) together with `previous > 0` and `poutcome != nonexistent`, which is contradictory. Rather than dropping these rows, `pdays` is set to missing for these records and an `is_inconsistent` flag is added — see [Feature Engineering](#feature-engineering).

## Approach
- A **skyline benchmark** was built on the leaking `duration` field using a Logistic Regression classifier, to gauge the ceiling of achievable performance.
- Five classifiers were trained and compared on the leakage-free feature set: **Logistic Regression, kNN, Decision Tree, XGBoost, LightGBM**.
- **ROC AUC** was used as the evaluation metric.
- **Feature engineering**: `LabelEncoder` for categoricals, `PolynomialFeatures` (degree=3) and `StandardScaler` for numerics, plus custom transformers for the `pdays` field (see below).
- All preprocessing and modeling steps were combined into a single scikit-learn **Pipeline**.
- Hyperparameters for the ensemble models (XGBoost, LightGBM) were tuned with both **RandomizedSearchCV** and **Hyperopt**.
- **SHAP** was used to interpret and explain the final model's predictions.

## Feature Engineering
Two custom scikit-learn-compatible transformers ([`custom_transformers.py`](./custom_transformers.py)) handle the `pdays`/`previous`/`poutcome` logic identified during EDA:

- **`PdaysFeatures`** — adds `pdays_never` (flag for clients never previously contacted) and `is_inconsistent` (flag for the contradictory records described above), and replaces the `999` filler code in `pdays` with `NaN` so it can be handled natively by models/imputers.
- **`IQRClipper`** — clips outliers in numeric columns to `[Q1 − k·IQR, Q3 + k·IQR]` bounds computed on the training data.

## Pipeline

```
Raw data
  └─▶ PdaysFeatures            (pdays_never, is_inconsistent flags; 999 → NaN)
        └─▶ IQRClipper          (outlier clipping on numeric features)
              └─▶ ColumnTransformer
                    ├─ categorical → LabelEncoder
                    └─ numeric     → StandardScaler → PolynomialFeatures (degree=3)
                          └─▶ Classifier (LogReg / kNN / Decision Tree / XGBoost / LightGBM)
```

*(Adjust this diagram if your actual pipeline step order or contents differ.)*

## Results

ROC AUC scores across models. `XGBoost + Hyperopt` was selected as the final model and additionally evaluated on the held-out **test** set.

| Classifier | Train | Validation | Test |
|---|---|---|---|
| Logistic Regression (benchmark, incl. `duration`) | 0.799 | 0.792 | — |
| kNN | 0.795 | 0.788 | — |
| Decision Tree | 0.792 | 0.796 | — |
| XGBoost + RandomizedSearchCV | 0.800 | 0.804 | — |
| **XGBoost + Hyperopt (final model)** | **0.844** | **0.810** | **0.823** |
| LightGBM + RandomizedSearchCV | 0.800 | 0.809 | — |
| LightGBM + Hyperopt | 0.859 | 0.801 | — |

> **Note:** in the original table draft, the `XGBoost + Hyperopt` column header used `colspan="2"` but the row underneath supplied three values (Train/Validation/**Test**). I've corrected the column span above — double-check the HTML table in your actual README renders 3 columns for that model if you keep the test score.

## Model Interpretation
SHAP values were used to explain feature contributions for the final model and validate that the socio-economic indicators (`euribor3m`, `nr.employed`, `emp.var.rate`) and the engineered `pdays`/`poutcome` features are among the key drivers of predictions.

## Repository Structure
```
├── 0_BankMarketingDataset_ML_EDA.ipynb        # EDA, hypotheses, data-quality checks
├── 1_BankMarketingDataset_ML_pipeline.ipynb   # feature engineering, pipeline, model training & tuning
├── 2_BankMarketingDataset_ML_analysis.ipynb   # results and SHAP analysis
├── custom_transformers.py                     # PdaysFeatures & IQRClipper transformers
├── visualization.py                            # plotting helpers used in the EDA notebook
├── bank-additional-full.csv                    # dataset (UCI Bank Marketing, not tracked in git)
├── BankMarket_model_XGB.joblib                  # final trained model
├── train_processed.parquet                      # processed training subset
├── val_processed.parquet                        # processed validation subset
├── test_processed.parquet                       # processed test subset
└── README.md
```

## How to Run
```bash
git clone https://github.com/<your-username>/BankMarketingData_Classification_DS_Project.git
cd BankMarketingData_Classification_DS_Project
pip install -r requirements.txt
```
Download `bank-additional-full.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) (or Kaggle) into the project root, then run the notebooks in order:

1. `0_BankMarketingDataset_ML_EDA.ipynb` — EDA and data-quality analysis (needs `visualization.py`)
2. `1_BankMarketingDataset_ML_pipeline.ipynb` — feature engineering, pipeline, model training and tuning (needs `custom_transformers.py`)
3. `2_BankMarketingDataset_ML_analysis.ipynb` — SHAP analysis (needs `BankMarket_model_XGB.joblib`, `train_processed.parquet`, `val_processed.parquet`, `test_processed.parquet`, and `custom_transformers.py`)

## Requirements
- pandas, numpy
- scikit-learn
- xgboost, lightgbm
- hyperopt
- shap
- statsmodels
- matplotlib, seaborn

## Author
*[ikonieva](https://github.com/ikonieva)*
