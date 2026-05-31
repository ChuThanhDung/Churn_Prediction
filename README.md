# Telco Customer Churn Dashboard

[Tiếng Việt](README.vi.md)

A reproducible machine learning portfolio project for predicting customer churn
from the IBM Telco Customer Churn dataset. The repository includes an analysis
notebook, three trained scikit-learn pipelines, a Streamlit dashboard, batch
prediction, tests, and CI.

## Live Demo

Deploy your fork on [Streamlit Community Cloud](https://share.streamlit.io/) and
add the resulting `*.streamlit.app` URL here.

## Business Problem

Customer churn prediction helps a telecom business identify customers who may
leave and prioritize retention actions. Recall matters because missing an
at-risk customer can cost more than contacting a customer who would have stayed.

## Model Results

All deployment models are evaluated on a stratified 60:40 validation split with
`random_state=42`. The dashboard defaults to Logistic Regression because it has
the strongest validation F1, AUC, and recall while remaining lightweight and
interpretable.

| Model | F1 | AUC | Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.6227 | 0.8418 | 0.8142 |
| SVM RBF | 0.6204 | 0.8299 | 0.8008 |
| Naive Bayes | 0.6034 | 0.8137 | 0.8115 |

## Features

- Overview KPIs and interactive exploratory data analysis.
- Comparison of Logistic Regression, SVM RBF, and mixed Naive Bayes pipelines.
- Single-customer churn probability prediction.
- Batch CSV upload with downloadable predictions.
- Reproducible training script and committed `joblib` artifacts.
- Pytest coverage and GitHub Actions CI on Python 3.12.

## Project Structure

```text
├── app.py                     # Streamlit overview
├── pages/                     # EDA, comparison, and prediction pages
├── src/                       # Cleaning, features, models, and inference
├── scripts/train_models.py    # Rebuild metrics, schema, and artifacts
├── artifacts/                 # Trained pipelines and public input contract
├── data/                      # IBM Telco Customer Churn CSV and attribution
├── notebooks/                 # Clean analysis notebook
└── tests/                     # Data, artifact, inference, and app smoke tests
```

## Run Locally

Use Python 3.12 to match Streamlit Community Cloud.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m scripts.train_models
pytest
streamlit run app.py
```

Open `http://localhost:8501`.

## Batch Prediction Contract

Uploaded CSV files must contain the 19 predictor columns documented in
[`artifacts/schema.json`](artifacts/schema.json). `customerID` and `Churn` are
optional passthrough columns. The downloaded result appends:

- `prediction`
- `churn_probability`
- `risk_level`
- `model`

Download a valid sample CSV directly from the Prediction page.

## Dataset

The repository keeps a local CSV so it runs immediately after cloning. The data
comes from IBM's archived
[Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d)
and its [raw CSV file](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
See [`data/README.md`](data/README.md) for attribution details.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from the repository, branch `main`, entrypoint `app.py`.
4. In Advanced settings, select Python `3.12`.
5. Deploy and add the generated URL to this README and the GitHub repository About section.

No secrets or external services are required.

## License

Project code is licensed under the [MIT License](LICENSE). Dataset attribution
and upstream licensing are documented separately.
