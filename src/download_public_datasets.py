"""
download_public_datasets.py - Utility to download and standardize external public benchmark datasets
(e.g., UCI Online Retail II / Instacart Market Basket subsets)
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

def download_uci_online_retail(target_dir=DATA_DIR):
    """
    Downloads UCI Online Retail dataset (541,909 transactions) and converts it to standard basket format.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "online_retail.zip")
    excel_path = os.path.join(target_dir, "Online Retail.xlsx")
    output_csv = os.path.join(target_dir, "uci_online_retail_baskets.csv")

    if os.path.exists(output_csv):
        print(f"Public dataset already converted: {output_csv}")
        return output_csv

    try:
        print(f"Attempting download from UCI Repository: {ONLINE_RETAIL_URL}...")
        urllib.request.urlretrieve(ONLINE_RETAIL_URL, zip_path)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        print("Parsing Online Retail transactions (this may take a minute)...")
        df = pd.read_excel(excel_path)
        # Clean data: drop cancellations and nulls
        df = df.dropna(subset=['CustomerID', 'Description'])
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        
        # Group into baskets
        baskets = df.groupby('InvoiceNo')['Description'].apply(lambda x: ','.join(sorted(list(set(x))))).reset_index()
        baskets.columns = ['transaction_id', 'items']
        baskets.to_csv(output_csv, index=False)
        print(f"Successfully processed {len(baskets):,} baskets to {output_csv}")
        return output_csv
    except Exception as e:
        print(f"Could not download or process external dataset automatically: {e}")
        print("Note: The project is fully equipped with our synthetic high-volume dataset in data/transactions.csv!")
        return None

if __name__ == "__main__":
    download_uci_online_retail()
