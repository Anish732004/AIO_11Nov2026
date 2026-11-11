# Reorder Point Lab

AI applications in reorder-point and inventory management (AIO, Session II exercise).

| Step | What the app does |
|---|---|
| 1. Demand forecasting | Synthetic daily demand with trend, annual and weekly seasonality, promotions (with post-promo dip), holidays and temperature. Random Forest and Gradient Boosting forecast **lead-time demand** (next L days), validated walk-forward against naïve and moving-average baselines. New data can be released day by day or uploaded as CSV; the model retrains and every forecast is logged and scored. |
| 2. Dynamic safety stock | *In progress:* analytic `Z·√(σ_D²·L + D̄²·σ_L²)` vs Monte Carlo. |
| 3. Reorder-point optimization | *In progress:* Bayesian optimization (Optuna) vs grid sweep. |

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

## Layout

- `app.py`: Streamlit dashboard
- `src/simulate.py`: synthetic demand generator (all drivers in `SimConfig`)
- `src/forecast.py`: features, walk-forward back-test, next-window forecast
- `src/data_io.py`: validation for uploaded rows
