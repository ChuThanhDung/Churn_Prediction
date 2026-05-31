# Dashboard Dự Đoán Telco Customer Churn

[English](README.md)

Đây là dự án portfolio machine learning có thể tái lập để dự đoán khách hàng rời
bỏ dịch vụ viễn thông. Repo gồm notebook phân tích, ba pipeline scikit-learn đã
huấn luyện, dashboard Streamlit, dự đoán CSV hàng loạt, kiểm thử và CI.

## Demo Trực Tuyến

Deploy bản fork trên [Streamlit Community Cloud](https://share.streamlit.io/) và
thêm URL `*.streamlit.app` vào đây.

## Kết Quả Mô Hình

Ba mô hình được đánh giá trên tập validation stratified 60:40 với
`random_state=42`. Dashboard mặc định dùng Logistic Regression vì đạt F1, AUC và
Recall tốt nhất, đồng thời nhẹ và dễ giải thích.

| Mô hình | F1 | AUC | Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.6227 | 0.8418 | 0.8142 |
| SVM RBF | 0.6204 | 0.8299 | 0.8008 |
| Naive Bayes | 0.6034 | 0.8137 | 0.8115 |

## Tính Năng

- KPI tổng quan và EDA tương tác.
- So sánh Logistic Regression, SVM RBF và mixed Naive Bayes.
- Dự đoán xác suất churn cho một khách hàng.
- Upload CSV và tải kết quả dự đoán hàng loạt.
- Script huấn luyện tái lập artifact `joblib`.
- Pytest và GitHub Actions CI với Python 3.12.

## Chạy Local

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m scripts.train_models
pytest
streamlit run app.py
```

Mở `http://localhost:8501`.

## Input CSV Hàng Loạt

CSV phải có đủ 19 predictor columns trong
[`artifacts/schema.json`](artifacts/schema.json). Hai cột `customerID` và `Churn`
là tùy chọn và được giữ lại trong output. Kết quả bổ sung:

- `prediction`
- `churn_probability`
- `risk_level`
- `model`

Trang Prediction có nút tải CSV mẫu hợp lệ.

## Dữ Liệu Và License

Dữ liệu đến từ [repo IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
Xem [`data/README.md`](data/README.md) để biết attribution. Phần code của dự án
dùng [MIT License](LICENSE); license này không thay thế license của dữ liệu nguồn.

## Deploy Streamlit Cloud

1. Push repo lên GitHub.
2. Mở [Streamlit Community Cloud](https://share.streamlit.io/).
3. Chọn repo, branch `main`, entrypoint `app.py`.
4. Trong Advanced settings, chọn Python `3.12`.
5. Deploy và cập nhật URL vào README cùng phần About của GitHub repo.
