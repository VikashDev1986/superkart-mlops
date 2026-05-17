"""
train.py — Model training, hyperparameter tuning, evaluation, HF registration.
Environment variables required:
  HF_TOKEN     : Hugging Face write token
  HF_USERNAME  : Hugging Face username  (default: vikashHugFace)
"""
import os, json, warnings, joblib
import pandas as pd
import numpy as np

os.environ.setdefault("CURL_CA_BUNDLE",     "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
import requests
_orig = requests.Session.request
def _no_verify(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig(self, method, url, **kwargs)
requests.Session.request = _no_verify

from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (BaggingRegressor, RandomForestRegressor,
                              AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from huggingface_hub import HfApi, create_repo

HF_TOKEN      = os.environ["HF_TOKEN"]
HF_USERNAME   = os.environ.get("HF_USERNAME", "vikashHugFace")
DATASET_REPO  = f"{HF_USERNAME}/superkart-sales-forecast"
MODEL_REPO    = f"{HF_USERNAME}/superkart-sales-forecast-model"
TARGET        = "Product_Store_Sales_Total"

# ── Load data ────────────────────────────────────────────────────────────
df_train = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN).to_pandas()
df_test  = load_dataset(DATASET_REPO, split="test",  token=HF_TOKEN).to_pandas()
for df in [df_train, df_test]:
    df.drop(columns=["Product_Id","Store_Id"], errors="ignore", inplace=True)

X_train, y_train = df_train.drop(columns=[TARGET]), df_train[TARGET]
X_test,  y_test  = df_test.drop(columns=[TARGET]),  df_test[TARGET]

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1))]), cat_cols),
])

# ── Baseline comparison ──────────────────────────────────────────────────
MODELS = {
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Bagging":           BaggingRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(random_state=42, n_jobs=-1),
    "AdaBoost":          AdaBoostRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost":           XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
}
PARAM_GRIDS = {
    "Decision Tree":     {"model__max_depth":[3,5,10,None], "model__min_samples_split":[2,5,10]},
    "Bagging":           {"model__n_estimators":[50,100,200], "model__max_samples":[0.6,0.8,1.0]},
    "Random Forest":     {"model__n_estimators":[100,200,300], "model__max_depth":[5,10,None]},
    "AdaBoost":          {"model__n_estimators":[50,100,200], "model__learning_rate":[0.1,0.5,1.0]},
    "Gradient Boosting": {"model__n_estimators":[100,200], "model__learning_rate":[0.05,0.1,0.2], "model__max_depth":[3,5]},
    "XGBoost":           {"model__n_estimators":[100,200], "model__learning_rate":[0.05,0.1,0.2], "model__max_depth":[3,5], "model__subsample":[0.8,1.0]},
}

baseline = {}
for name, est in MODELS.items():
    pipe   = Pipeline([("prep", preprocessor), ("model", est)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
    baseline[name] = -scores.mean()
    print(f"  {name:<22} CV-RMSE = {baseline[name]:,.2f}")

best_name = min(baseline, key=baseline.get)
print(f"\nBest baseline: {best_name}  RMSE={baseline[best_name]:,.2f}")

# ── GridSearchCV ─────────────────────────────────────────────────────────
best_pipe = Pipeline([("prep", preprocessor), ("model", MODELS[best_name])])
gs = GridSearchCV(best_pipe, PARAM_GRIDS[best_name], cv=5,
                  scoring="neg_root_mean_squared_error", n_jobs=-1,
                  verbose=1, refit=True)
gs.fit(X_train, y_train)
best_tuned_rmse = -gs.best_score_
print(f"Tuned CV-RMSE: {best_tuned_rmse:,.2f}")
print(f"Best params  : {gs.best_params_}")

# ── Test-set evaluation ───────────────────────────────────────────────────
y_pred = gs.best_estimator_.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
mape   = np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100
print(f"Test RMSE={rmse:,.2f}  MAE={mae:,.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")

# ── Save + register ───────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(gs.best_estimator_, "model/best_model.pkl")
meta = {"model_type": best_name, "best_params": gs.best_params_,
        "cv_rmse": round(best_tuned_rmse,4), "test_rmse": round(rmse,4),
        "test_mae": round(mae,4), "test_r2": round(r2,4),
        "test_mape_pct": round(mape,4), "features": num_cols+cat_cols,
        "target": TARGET}
with open("model/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

api = HfApi(token=HF_TOKEN)
create_repo(MODEL_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
for fname in ["best_model.pkl", "metadata.json"]:
    api.upload_file(path_or_fileobj=f"model/{fname}", path_in_repo=fname,
                    repo_id=MODEL_REPO, repo_type="model")
    print(f"Uploaded model/{fname}")
print(f"Model registered at: https://huggingface.co/{MODEL_REPO}")
