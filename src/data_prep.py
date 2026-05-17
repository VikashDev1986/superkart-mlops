"""
data_prep.py — Data loading, cleaning, train/test split, and HF Hub upload.
Environment variables required:
  HF_TOKEN      : Hugging Face write token
  HF_USERNAME   : Hugging Face username  (default: vikashHugFace)
"""
import os, warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict

os.environ.setdefault("CURL_CA_BUNDLE",     "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
warnings.filterwarnings("ignore", message="Unverified HTTPS request")
import requests
_orig = requests.Session.request
def _no_verify(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig(self, method, url, **kwargs)
requests.Session.request = _no_verify

HF_TOKEN     = os.environ["HF_TOKEN"]
HF_USERNAME  = os.environ.get("HF_USERNAME", "vikashHugFace")
DATASET_NAME = "superkart-sales-forecast"
REPO_ID      = f"{HF_USERNAME}/{DATASET_NAME}"
TARGET       = "Product_Store_Sales_Total"
SUGAR_MAP    = {"LF": "Low Sugar", "low fat": "Low Sugar",
                "Low Fat": "Low Sugar", "reg": "Regular"}

print(f"Loading dataset from HF Hub: {REPO_ID}")
df = load_dataset(REPO_ID, split="train", token=HF_TOKEN).to_pandas()
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

# Clean
df = df.drop(columns=["Product_Id", "Store_Id"], errors="ignore")
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace(SUGAR_MAP)
num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
print(f"Cleaned: {df.isnull().sum().sum()} nulls remaining")

# Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train: {len(df_train):,}  |  Test: {len(df_test):,}")

# Save locally
os.makedirs("data", exist_ok=True)
df_train.to_csv("data/train.csv", index=False)
df_test.to_csv("data/test.csv",   index=False)
print("Saved data/train.csv and data/test.csv")

# Upload splits to HF Hub
splits = DatasetDict({
    "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
    "test" : Dataset.from_pandas(df_test.reset_index(drop=True)),
})
splits.push_to_hub(REPO_ID, token=HF_TOKEN)
print(f"Uploaded train/test splits to: https://huggingface.co/datasets/{REPO_ID}")
