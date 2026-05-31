"""Model definitions for the Telco Customer Churn dashboard."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data import NUMERIC_COLUMNS
from src.features import build_naive_bayes_preprocessor, build_standard_preprocessor

MODEL_NAMES = ("logistic_regression", "svm_rbf", "naive_bayes")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "SVM RBF",
    "naive_bayes": "Naive Bayes",
}
DEFAULT_MODEL = "logistic_regression"


class MixedGaussianBernoulliNB(BaseEstimator, ClassifierMixin):
    """Use Gaussian NB for numeric values and Bernoulli NB for encoded categories."""

    def __init__(self, n_numeric: int):
        self.n_numeric = n_numeric

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MixedGaussianBernoulliNB":
        values = np.asarray(X)
        self.classes_ = np.unique(y)
        self.gaussian_nb_ = GaussianNB()
        self.bernoulli_nb_ = BernoulliNB()
        self.gaussian_nb_.fit(values[:, : self.n_numeric], y)
        self.bernoulli_nb_.fit(values[:, self.n_numeric :], y)
        return self

    def _joint_log_proba(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X)
        gaussian_joint = self.gaussian_nb_.predict_joint_log_proba(
            values[:, : self.n_numeric]
        )
        bernoulli_joint = self.bernoulli_nb_.predict_joint_log_proba(
            values[:, self.n_numeric :]
        )
        return gaussian_joint + bernoulli_joint - self.bernoulli_nb_.class_log_prior_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_proba(X)
        log_norm = np.logaddexp.reduce(joint, axis=1)
        return np.exp(joint - log_norm[:, None])

    def predict(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_proba(X)
        return self.classes_[np.argmax(joint, axis=1)]


def build_model(model_name: str) -> Pipeline:
    """Create a fitted-ready pipeline using the selected notebook parameters."""
    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("preprocess", build_standard_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=1,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == "svm_rbf":
        return Pipeline(
            steps=[
                ("preprocess", build_standard_preprocessor()),
                (
                    "model",
                    SVC(
                        C=0.1,
                        gamma=0.1,
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        cache_size=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == "naive_bayes":
        return Pipeline(
            steps=[
                ("preprocess", build_naive_bayes_preprocessor()),
                ("model", MixedGaussianBernoulliNB(n_numeric=len(NUMERIC_COLUMNS))),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")
