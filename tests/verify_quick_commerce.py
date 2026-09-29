"""
verify_quick_commerce.py - Multi-Unit Comprehensive Test Suite
Validates Unit 1 (Frontend assets), Unit 2 (Retail business logic), and Unit 3 (ML Multi-Item Rules).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_recommender import HybridRecommender
from src.product_catalog import (
    PRODUCT_IMAGES,
    PRODUCT_SPECS,
    get_product_image,
    get_product_spec,
    get_aisle_info
)

def run_tests():
    print("================================================================")
    print("UNIT 1 TEST: Frontend Assets & Verified Photo Coverage")
    print("================================================================")
    assert len(PRODUCT_IMAGES) == 74, f"Expected 74 product images, found {len(PRODUCT_IMAGES)}"
    assert len(PRODUCT_SPECS) == 74, f"Expected 74 product specs, found {len(PRODUCT_SPECS)}"
    for pid, url in PRODUCT_IMAGES.items():
        assert url.startswith("https://images.unsplash.com/"), f"Invalid CDN URL for {pid}"
    print("✅ Unit 1 Passed: All 74 products have verified high-resolution CDN images & specs.")

    print("\n================================================================")
    print("UNIT 3 TEST: ML Multi-Item Antecedent Association Rule Querying")
    print("================================================================")
    rec = HybridRecommender(data_dir="data")
    rec.fit(min_support=0.015, min_lift=1.2)
    
    # Query multi-item antecedent rules for Penne + Marinara
    cart_test = ["PROD_001", "PROD_004"]
    multi_rules = rec.miner.get_rules_for_cart_subsets(cart_test, top_n=10)
    assert len(multi_rules) > 0, "No multi-item rules extracted for cart!"
    
    # Check that multi-item antecedents exist
    has_len_2 = any(r['antecedent_len'] == 2 for r in multi_rules)
    assert has_len_2, "Failed to match length-2 antecedent rule!"
    print(f"✅ Unit 3 Passed: Successfully extracted {len(multi_rules)} cart-subset rules with multi-item antecedents.")
    top_multi = [r for r in multi_rules if r['antecedent_len'] == 2][0]
    print(f"   Top Multi-Item Rule: {{{top_multi['antecedent_names']}}} => {top_multi['consequent_name']} (Lift: {top_multi['lift']:.2f}x, Conf: {top_multi['confidence']*100:.1f}%)")

    print("\n================================================================")
    print("UNIT 2 TEST: Retail Business Logic & Substitute Suppression")
    print("================================================================")
    # Test recommendations WITH substitute suppression filter
    recs_filtered = rec.recommend_for_cart(cart_test, top_n=5, filter_substitutes=True)
    rec_names_filtered = [r['product_name'] for r in recs_filtered]
    
    # Verify that competing dry pastas (Spaghetti, Fettuccine) are NOT recommended when Penne is in cart
    assert "Organic Spaghetti Pasta" not in rec_names_filtered, "Spaghetti substitute was incorrectly recommended!"
    assert "Fettuccine Egg Pasta" not in rec_names_filtered, "Fettuccine substitute was incorrectly recommended!"
    
    print("✅ Unit 2 Passed: Direct substitute cannibalization successfully suppressed.")
    print("   Recommended Cross-Category Complements:")
    for i, r in enumerate(recs_filtered, 1):
        print(f"   {i}. {r['product_name']} ({r['category']}) - ${r['price']:.2f} | Lift: {r['transaction_lift']:.1f}x")
        print(f"      Rationale: {r['explanation']}")

    print("\n================================================================")
    print("ALL 3 UNITS VALIDATED: FRONTEND, BUSINESS LOGIC, AND ML ALGO ✅")
    print("================================================================")

if __name__ == "__main__":
    run_tests()
