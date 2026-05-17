"""
SuperKart Sales Forecast — Gradio inference app.
Loads the best-tuned sklearn pipeline from Hugging Face Model Hub
and serves it as an interactive web form.
"""
import os, warnings
import joblib
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download

import requests
_orig = requests.Session.request
def _no_verify(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig(self, method, url, **kwargs)
requests.Session.request = _no_verify

# ── Load model from HF Hub ────────────────────────────────────────────────
MODEL_REPO = "vikashHugFace/superkart-sales-forecast-model"
model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pkl")
model      = joblib.load(model_path)
print(f"Model loaded from {MODEL_REPO}")

# ── Inference function ────────────────────────────────────────────────────
def predict_sales(
    product_weight: float,
    product_sugar_content: str,
    product_allocated_area: float,
    product_type: str,
    product_mrp: float,
    store_establishment_year: int,
    store_size: str,
    store_location_city_type: str,
    store_type: str,
) -> str:
    """Build a one-row DataFrame from user inputs and return the prediction."""
    input_df = pd.DataFrame([{
        "Product_Weight":           product_weight,
        "Product_Sugar_Content":    product_sugar_content,
        "Product_Allocated_Area":   product_allocated_area,
        "Product_Type":             product_type,
        "Product_MRP":              product_mrp,
        "Store_Establishment_Year": store_establishment_year,
        "Store_Size":               store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type":               store_type,
    }])
    prediction = model.predict(input_df)[0]
    return f"{prediction:,.2f}"

# ── Gradio UI ─────────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_sales,
    inputs=[
        gr.Number(label="Product Weight (kg)",            value=12.0),
        gr.Dropdown(label="Product Sugar Content",
                    choices=["Low Sugar", "Regular", "No Sugar"],
                    value="Regular"),
        gr.Number(label="Product Allocated Area (ratio)", value=0.05),
        gr.Dropdown(label="Product Type",
                    choices=["Dairy","Soft Drinks","Meat","Fruits and Vegetables",
                             "Household","Baking Goods","Snack Foods","Frozen Foods",
                             "Breakfast","Health and Hygiene","Hard Drinks","Canned",
                             "Breads","Starchy Foods","Others","Seafood"],
                    value="Dairy"),
        gr.Number(label="Product MRP (₹)",                value=150.0),
        gr.Number(label="Store Establishment Year",        value=2000, precision=0),
        gr.Dropdown(label="Store Size",
                    choices=["High","Medium","Small"], value="Medium"),
        gr.Dropdown(label="Store Location City Type",
                    choices=["Tier 1","Tier 2","Tier 3"], value="Tier 2"),
        gr.Dropdown(label="Store Type",
                    choices=["Supermarket Type1","Supermarket Type2",
                             "Supermarket Type3","Grocery Store"],
                    value="Supermarket Type1"),
    ],
    outputs=gr.Textbox(label="Predicted Total Sales (₹)"),
    title="🛒 SuperKart Sales Forecast",
    description="Enter product and store attributes to predict total revenue.",
    examples=[
        [12.66,"Low Sugar",0.027,"Frozen Foods",117.08,2009,"Medium","Tier 2","Supermarket Type2"],
        [16.54,"Low Sugar",0.144,"Dairy",171.43,1999,"Medium","Tier 1","Grocery Store"],
        [14.28,"Regular",0.031,"Canned",162.08,1987,"High","Tier 2","Supermarket Type1"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
