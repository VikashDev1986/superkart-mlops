---
    tags:
      - regression
      - sales-forecasting
      - sklearn
    metrics:
      - rmse
      - mae
      - r2
    ---

    # SuperKart Sales Forecast Model

    **Algorithm:** Random Forest  
    **Task:** Regression — predict `Product_Store_Sales_Total`

    ## Performance (held-out test set)

    | Metric | Value |
    |--------|-------|
    | RMSE   | 279.80 |
    | MAE    | 113.79 |
    | R²     | 0.9314 |
    | MAPE   | 4.06 % |

    ## Best Hyperparameters

    ```json
    {
  "model__max_depth": 10,
  "model__min_samples_split": 5,
  "model__n_estimators": 300
}
    ```

    ## Usage

    ```python
    import joblib
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="vikashHugFace/superkart-sales-forecast-model", filename="best_model.pkl")
    model = joblib.load(path)
    predictions = model.predict(X_new)
    ```