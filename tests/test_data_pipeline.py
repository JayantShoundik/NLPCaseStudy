"""
test_data_pipeline.py - Unit tests for data generation, schema integrity, and distribution bounds.
"""

import os
import sys
import unittest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

class TestDataPipeline(unittest.TestCase):
    def test_products_file_exists_and_valid(self):
        path = os.path.join(DATA_DIR, "products.csv")
        self.assertTrue(os.path.exists(path), "products.csv should exist")
        df = pd.read_csv(path)
        self.assertGreaterEqual(len(df), 60, "Expected at least 60 products")
        self.assertTrue(df['id'].is_unique, "Product IDs must be unique")
        self.assertEqual(df['id'].isnull().sum(), 0, "No null product IDs allowed")
        self.assertTrue((df['price'] > 0).all(), "All products must have positive price")
        self.assertIn('description', df.columns, "Products must have descriptions")

    def test_transactions_schema_and_distribution(self):
        path = os.path.join(DATA_DIR, "train_transactions.csv")
        self.assertTrue(os.path.exists(path), "train_transactions.csv should exist")
        df = pd.read_csv(path)
        self.assertGreaterEqual(len(df), 10000, "Expected at least 10,000 train transactions")
        self.assertTrue(df['transaction_id'].is_unique, "Transaction IDs must be unique")
        self.assertEqual(df['items'].isnull().sum(), 0, "No null baskets allowed")
        # Check basket sizes
        basket_sizes = df['items'].apply(lambda x: len(str(x).split(',')))
        self.assertTrue((basket_sizes >= 2).all(), "All baskets must contain at least 2 items")
        self.assertLessEqual(basket_sizes.max(), 25, "Basket sizes should be realistic")

    def test_reviews_sentiment_and_ratings(self):
        path = os.path.join(DATA_DIR, "reviews.csv")
        self.assertTrue(os.path.exists(path), "reviews.csv should exist")
        df = pd.read_csv(path)
        self.assertGreaterEqual(len(df), 5000, "Expected at least 5,000 reviews")
        self.assertTrue((df['rating'] >= 1).all() and (df['rating'] <= 5).all(), "Ratings must be between 1 and 5")
        self.assertEqual(df['review_text'].isnull().sum(), 0, "No null review text allowed")

if __name__ == '__main__':
    unittest.main()
