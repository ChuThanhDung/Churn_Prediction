"""Single-customer and batch churn prediction page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import CATEGORY_OPTIONS, InputValidationError, clean_data, load_raw_data
from src.inference import predict_batch, predict_customer
from src.models import DEFAULT_MODEL, MODEL_LABELS

st.set_page_config(
    page_title="Prediction | Telco Churn",
    page_icon="🎯",
    layout="wide",
)


@st.cache_data
def sample_batch_csv() -> bytes:
    """Provide a valid starter file for the batch prediction interface."""
    sample = clean_data(load_raw_data()).drop(columns=["Churn"]).head(5)
    return sample.to_csv(index=False).encode("utf-8")


def option(column: str, default: str | None = None) -> str:
    """Render a categorical selector with a consistent default."""
    values = CATEGORY_OPTIONS[column]
    index = values.index(default) if default in values else 0
    return st.selectbox(column, values, index=index)


st.title("Churn Prediction | Dự đoán churn")
st.caption("Choose a trained pipeline and score one customer or an uploaded CSV file.")

model_name = st.sidebar.selectbox(
    "Model | Mô hình",
    options=list(MODEL_LABELS),
    format_func=lambda name: MODEL_LABELS[name],
    index=list(MODEL_LABELS).index(DEFAULT_MODEL),
)

single_tab, batch_tab = st.tabs(
    ["Single customer | Một khách hàng", "Batch CSV | Dự đoán hàng loạt"]
)

with single_tab:
    with st.form("customer-form"):
        st.subheader("Customer profile | Thông tin khách hàng")
        first, second, third = st.columns(3)
        with first:
            gender = option("gender", "Female")
            senior_citizen = st.selectbox("SeniorCitizen", [0, 1])
            partner = option("Partner", "No")
            dependents = option("Dependents", "No")
            tenure = st.number_input("tenure", min_value=0, max_value=100, value=12)
            phone_service = option("PhoneService", "Yes")
            multiple_lines = option("MultipleLines", "No")
        with second:
            internet_service = option("InternetService", "Fiber optic")
            online_security = option("OnlineSecurity", "No")
            online_backup = option("OnlineBackup", "No")
            device_protection = option("DeviceProtection", "No")
            tech_support = option("TechSupport", "No")
            streaming_tv = option("StreamingTV", "No")
            streaming_movies = option("StreamingMovies", "No")
        with third:
            contract = option("Contract", "Month-to-month")
            paperless_billing = option("PaperlessBilling", "Yes")
            payment_method = option("PaymentMethod", "Electronic check")
            monthly_charges = st.number_input(
                "MonthlyCharges",
                min_value=0.0,
                value=75.0,
                step=0.05,
            )
            total_charges = st.number_input(
                "TotalCharges",
                min_value=0.0,
                value=900.0,
                step=0.05,
            )
        submitted = st.form_submit_button("Predict churn risk | Dự đoán")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        try:
            prediction = predict_customer(customer, model_name)
        except (InputValidationError, FileNotFoundError) as error:
            st.error(str(error))
        else:
            first, second, third = st.columns(3)
            first.metric("Prediction | Dự đoán", prediction["prediction"])
            second.metric(
                "Churn probability | Xác suất",
                f"{prediction['churn_probability']:.1%}",
            )
            third.metric("Risk level | Mức rủi ro", prediction["risk_level"])
            st.caption(f"Model: {prediction['model']}")

with batch_tab:
    st.download_button(
        "Download sample CSV | Tải CSV mẫu",
        data=sample_batch_csv(),
        file_name="telco_batch_sample.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader(
        "Upload customer CSV | Tải tệp khách hàng",
        type=["csv"],
    )
    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)
        st.write(f"Rows | Số dòng: {len(uploaded_data):,}")
        try:
            batch_result = predict_batch(uploaded_data, model_name)
        except (InputValidationError, FileNotFoundError) as error:
            st.error(str(error))
        else:
            st.dataframe(batch_result.head(100), use_container_width=True)
            st.download_button(
                "Download predictions | Tải kết quả",
                data=batch_result.to_csv(index=False).encode("utf-8"),
                file_name="telco_churn_predictions.csv",
                mime="text/csv",
            )
