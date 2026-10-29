# Predicting serious delinquency in consumer credit

Group 4 | eHA Academy Data Science Capstone | FinTech Risk and Fraud track

Members: Ahmed Yunus

## Problem

A lender's credit risk team has limited staff and cannot contact every borrower about late payments. This project ranks borrowers by their risk of becoming 90 or more days past due within 2 years, so the team can start with the highest scores.

Targets set before modelling: recall of at least 60% and precision of at least 20%.

## Data

Kaggle competition Give Me Some Credit, file cs-training.csv: 150,000 borrowers, 10 inputs and a target. After cleaning, 149,390 borrowers remain and 6.70% became delinquent. The files are not in this repo. Download them from https://www.kaggle.com/c/GiveMeSomeCredit/data (a Kaggle login and accepted competition rules may be needed) and put cs-training.csv in a folder named Dataset.

## Result

Final model: gradient boosting on 9 inputs, with a threshold of 0.134 fixed on the training part. The test part was used once.

| Measure | Test part |
|---|---|
| Recall | 0.619 (95% interval 0.598 to 0.641) |
| Precision | 0.322 (95% interval 0.309 to 0.338) |
| PR-AUC | 0.406 (random guessing: 0.067) |
| ROC-AUC | 0.861 |

The model flags 12.9% of borrowers. The lower end of the recall interval is just under the 60% target.

## Files

| File | What it holds |
|---|---|
| 01_eda.ipynb | Modules 1 to 5: loading, cleaning, new columns, exploration, statistics and the train and test split |
| 02_ml.ipynb | Modules 6 and 7: cleaning stage B, baseline, gradient boosting, input tests, error analysis and the final test |
| demo/app.py | Streamlit demo that scores one borrower |
| demo/model.pkl | Saved model, with the threshold and the training caps |
| processed/experiment_log.csv, processed/ablation.csv | Scores of every model run |
| requirements.txt | Pinned libraries for the demo |

## Run the demo

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run demo/app.py
```

To rerun the notebooks, also install matplotlib, seaborn, jupyter, ipykernel and kagglehub, and run 01_eda.ipynb before 02_ml.ipynb.

## Limits

- The data source does not state a country or a year.
- The test part is a random 20% of the same file, so the model was not tested on a later period or in another market.
- Age was left out of the final model because it is a sensitive input in lending. Recall still falls with age, from 69.3% for ages 18 to 29 to 53.5% for 60 and over.
- About a third of delinquent borrowers had no late payments and the model mostly misses them.
- The demo is for demonstration only and not for real lending decisions.
With Docker instead:

```
docker compose up --build
```

Live demo: https://capstone-group4-credit-risk-qxdkf8mhigsaocvvuvccdm.streamlit.app/
