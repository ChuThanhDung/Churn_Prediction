"""Reproducible model metrics comparison page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"

st.set_page_config(
    page_title="Model Comparison | Telco Churn",
    page_icon="📋",
    layout="wide",
)

st.title("Model Comparison | So sánh mô hình")
st.caption("Stratified 60:40 validation split with random_state=42.")

if not METRICS_PATH.exists():
    st.error("Metrics are missing. Run: python -m scripts.train_models")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
metric_frame = pd.DataFrame(
    [
        {
            "Model": values["display_name"],
            "Accuracy": values["accuracy"],
            "Balanced Accuracy": values["balanced_accuracy"],
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1"],
            "AUC": values["auc"],
        }
        for values in metrics.values()
    ]
).set_index("Model")

st.dataframe(metric_frame.style.format("{:.4f}"), use_container_width=True)

st.subheader("Churn-focused metrics | Chỉ số trọng tâm")
st.bar_chart(metric_frame[["F1", "AUC", "Recall"]])

st.success(
    "Logistic Regression is the recommended default: it has the best validation "
    "F1, AUC and recall while remaining lightweight and interpretable. "
    "| Logistic Regression là lựa chọn mặc định vì hiệu quả tốt, nhẹ và dễ giải thích."
)

with st.expander("How to reproduce | Cách tái lập"):
    st.code("python -m scripts.train_models", language="bash")
