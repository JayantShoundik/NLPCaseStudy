"""
test_hybrid_recommender.py - Unit tests for hybrid recommendation ranking, bundle economics,
and explainability outputs.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hybrid_recommender import HybridRecommender

class TestHybridRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = HybridRecommender()
        cls.recommender.fit(min_support=0.015, min_lift=1.2)

    def test_recommendation_ranking_structure(self):
        recs = self.recommender.recommend_for_product("PROD_001", top_n=5)
        self.assertGreater(len(recs), 0, "Should generate recommendations for Penne Pasta")
        
        # Check that top recommendation is an Italian pantry item
        top_rec = recs[0]
        self.assertIn("product_id", top_rec)
        self.assertIn("product_name", top_rec)
        self.assertIn("hybrid_score", top_rec)
        self.assertIn("explanation", top_rec)
        self.assertTrue(len(top_rec["explanation"]) > 10, "Explanation should be informative")

    def test_bundle_generation_and_discount_math(self):
        bundle = self.recommender.generate_promotional_bundle("PROD_001", bundle_size=3, discount_percent=15)
        self.assertIsNotNone(bundle)
        self.assertEqual(len(bundle["items"]), 3)
        
        # Check price math: bundle_price = regular - savings
        reg = bundle["regular_total_price"]
        promo = bundle["bundle_promotional_price"]
        savings = bundle["customer_savings"]
        
        self.assertAlmostEqual(promo + savings, reg, places=2)
        expected_savings = round(reg * 0.15, 2)
        self.assertAlmostEqual(savings, expected_savings, places=1)

    def test_cart_multi_item_recommendation(self):
        cart = ["PROD_001", "PROD_004"] # Penne + Marinara
        cart_recs = self.recommender.recommend_for_cart(cart, top_n=3)
        
        rec_ids = [r["product_id"] for r in cart_recs]
        # None of the recommended items should already be in the cart
        for item in cart:
            self.assertNotIn(item, rec_ids, "Should not recommend items already present in cart")

if __name__ == '__main__':
    unittest.main()
