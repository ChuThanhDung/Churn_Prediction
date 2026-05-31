"""Interactive exploratory data analysis page."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data import clean_data, load_raw_data

st.set_page_config(page_title="EDA | Telco Churn", page_icon="📈", layout="wide")


@st.cache_data
def get_clean_data() -> pd.DataFrame:
    return clean_data(load_raw_data())


data = get_clean_data()
st.title("Exploratory Data Analysis | Phân tích dữ liệu")
st.caption("Key churn patterns from the IBM Telco Customer Churn dataset.")

category = st.selectbox(
    "Compare churn by category | So sánh churn theo nhóm",
    ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity"],
)

category_rate = (
    data.groupby(category)["Churn"]
    .apply(lambda values: (values == "Yes").mean())
    .sort_values(ascending=False)
    .rename("Churn rate")
)

left, right = st.columns(2)
with left:
    st.subheader(f"Churn rate by {category}")
    st.bar_chart(category_rate)
with right:
    st.subheader("Class distribution | Phân bố nhãn")
    st.bar_chart(data["Churn"].value_counts())

st.subheader("Numeric distributions | Phân bố biến số")
numeric_feature = st.radio(
    "Feature | Thuộc tính",
    ["tenure", "MonthlyCharges", "TotalCharges"],
    horizontal=True,
)
figure, axis = plt.subplots(figsize=(9, 4))
sns.histplot(
    data=data,
    x=numeric_feature,
    hue="Churn",
    bins=30,
    stat="density",
    common_norm=False,
    alpha=0.4,
    ax=axis,
)
axis.set_title(f"{numeric_feature} distribution by churn status")
st.pyplot(figure)

with st.expander("Clean dataset preview | Xem dữ liệu đã làm sạch"):
    st.dataframe(data.head(50), use_container_width=True)
