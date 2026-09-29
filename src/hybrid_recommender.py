"""
hybrid_recommender.py - Hybrid Association Mining + NLP Semantic Recommender & Bundle Generator
Combines transactional Lift from Apriori/FP-Growth with NLP textual similarity and review sentiment
to generate explainable cross-sell recommendations and high-converting promotional bundles.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.market_basket_miner import MarketBasketMiner
from src.nlp_engine import NLPEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

class HybridRecommender:
    def __init__(self, data_dir=DATA_DIR, weight_lift=0.55, weight_nlp=0.30, weight_sentiment=0.15):
        self.data_dir = data_dir
        self.w_lift = weight_lift
        self.w_nlp = weight_nlp
        self.w_sent = weight_sentiment
        
        self.miner = MarketBasketMiner(data_dir=data_dir)
        self.nlp = NLPEngine(data_dir=data_dir)
        self.is_fitted = False

    def fit(self, min_support=0.015, min_lift=1.2):
        """Fits both Association Miner and NLP Engine."""
        print("\n--- Initializing Hybrid Recommendation Engine ---")
        self.nlp.fit_all()
        self.miner.fit_all(min_support=min_support, min_lift=min_lift)
        self.is_fitted = True
        print("Hybrid Recommender successfully trained and ready.")
        return self

    def recommend_for_product(self, product_id, top_n=5):
        """
        Generates top complementary product recommendations for a given seed item.
        Blends transaction rule Lift + NLP description similarity + Review sentiment affinity.
        """
        if not self.is_fitted:
            self.fit()

        prod_info = self.miner.products_lookup.get(product_id)
        if not prod_info:
            return []

        # 1. Transaction Association Rules
        assoc_rules = self.miner.get_rules_for_product(product_id, top_n=30)
        assoc_dict = {r['consequent_id']: r for r in assoc_rules}

        # 2. NLP Semantic Similarities
        nlp_similar = self.nlp.get_top_nlp_similar_products(product_id, top_n=30)
        nlp_dict = {s['product_id']: s for s in nlp_similar}

        # 3. Review Sentiment & Pairings
        sentiment_info = self.nlp.product_sentiment
        pairing_insights = self.nlp.pairing_insights.get(product_id, {})
        top_pairing_words = list(pairing_insights.get('top_pairings', {}).keys())

        # Collect candidate IDs (excluding seed itself)
        candidates = set(assoc_dict.keys()).union(set(nlp_dict.keys()))
        candidates.discard(product_id)

        scored_candidates = []
        max_lift = max([r['lift'] for r in assoc_rules]) if assoc_rules else 1.0

        for cand_id in candidates:
            cand_info = self.miner.products_lookup.get(cand_id, {})
            cand_name = cand_info.get('name', cand_id)

            # Lift Score normalized to [0, 1]
            raw_lift = assoc_dict.get(cand_id, {}).get('lift', 0.0)
            norm_lift = min(raw_lift / max_lift, 1.0) if max_lift > 0 else 0.0
            confidence = assoc_dict.get(cand_id, {}).get('confidence', 0.0)
            support = assoc_dict.get(cand_id, {}).get('support', 0.0)

            # NLP Similarity score [0, 1]
            nlp_sim = self.nlp.get_semantic_similarity(product_id, cand_id)

            # Review sentiment boost
            cand_sent = sentiment_info.get(cand_id, {})
            mean_compound = cand_sent.get('mean_sentiment', 0.0)
            # Normalize [-1, 1] to [0, 1]
            norm_sentiment = (mean_compound + 1.0) / 2.0

            # Pairing keyword bonus if candidate name appears in seed's review pairings
            keyword_bonus = 0.0
            for word in top_pairing_words:
                if word in cand_name.lower():
                    keyword_bonus = 0.15
                    break

            # Hybrid Score calculation
            if raw_lift > 0:
                hybrid_score = (self.w_lift * norm_lift) + (self.w_nlp * nlp_sim) + (self.w_sent * norm_sentiment) + keyword_bonus
            else:
                # Fallback purely to NLP & Sentiment when no transaction rule exists
                hybrid_score = (0.70 * nlp_sim) + (0.30 * norm_sentiment) + keyword_bonus

            # Generate natural language explainability string
            explanation = self._build_explanation(
                seed_name=prod_info['name'],
                target_name=cand_name,
                raw_lift=raw_lift,
                confidence=confidence,
                nlp_sim=nlp_sim,
                review_quote=cand_sent.get('sentiment_label', 'Positive'),
                sample_quote=pairing_insights.get('sample_quote', '')
            )

            scored_candidates.append({
                "product_id": cand_id,
                "product_name": cand_name,
                "category": cand_info.get('category', 'Unknown'),
                "price": float(cand_info.get('price', 0.0)),
                "hybrid_score": round(float(hybrid_score), 4),
                "transaction_lift": round(float(raw_lift), 2),
                "transaction_confidence": round(float(confidence), 3),
                "transaction_support": round(float(support), 4),
                "nlp_semantic_similarity": round(float(nlp_sim), 4),
                "customer_sentiment": cand_sent.get('sentiment_label', 'Positive'),
                "customer_rating": cand_sent.get('mean_rating', 4.5),
                "explanation": explanation
            })

        scored_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return scored_candidates[:top_n]

    def _build_explanation(self, seed_name, target_name, raw_lift, confidence, nlp_sim, review_quote, sample_quote):
        """Generates clear, persuasive natural language reasoning for the recommendation."""
        reasons = []
        if raw_lift >= 2.0:
            reasons.append(f"Shoppers are {raw_lift:.1f}x more likely to purchase this together (Confidence: {confidence*100:.1f}%)")
        elif raw_lift > 1.0:
            reasons.append(f"Frequently co-purchased in customer carts (Confidence: {confidence*100:.1f}%)")

        if nlp_sim >= 0.20:
            reasons.append("High recipe & ingredient complementarity based on product descriptions")

        if sample_quote and len(sample_quote) > 10:
            snippet = sample_quote[:120] + "..." if len(sample_quote) > 120 else sample_quote
            reasons.append(f"Top customer review: \"{snippet}\"")

        return " • ".join(reasons) if reasons else "Complementary culinary pairing recommended based on purchase trends."

    def get_cart_gastronomic_health(self, cart_product_ids):
        """
        AI Culinary Basket Health & Balance Diagnostic:
        Evaluates active shopping cart against culinary gastronomy taxonomy:
        [1. Base Carb] + [2. Sauce/Dressing] + [3. Fresh Produce/Aromatic] + [4. Protein/Dairy].
        Calculates balance score and recommends the missing culinary link.
        """
        if not cart_product_ids:
            return {
                "score": 0,
                "label": "Empty Basket",
                "missing_roles": ["Carbohydrate Base", "Sauce / Dressing", "Fresh Produce", "Protein / Dairy"],
                "nudge": "Add a foundational staple (pasta, bread, rice, tortillas) to start crafting a chef-balanced meal!"
            }

        role_map = {
            "PROD_001": "Base Carb", "PROD_002": "Base Carb", "PROD_003": "Base Carb", "PROD_013": "Base Carb", "PROD_016": "Base Carb", "PROD_017": "Base Carb", "PROD_020": "Base Carb", "PROD_035": "Base Carb", "PROD_036": "Base Carb", "PROD_045": "Base Carb", "PROD_046": "Base Carb",
            "PROD_004": "Sauce / Dressing", "PROD_005": "Sauce / Dressing", "PROD_006": "Sauce / Dressing", "PROD_014": "Sauce / Dressing", "PROD_019": "Sauce / Dressing", "PROD_037": "Sauce / Dressing", "PROD_038": "Sauce / Dressing", "PROD_041": "Sauce / Dressing", "PROD_047": "Sauce / Dressing", "PROD_048": "Sauce / Dressing", "PROD_049": "Sauce / Dressing", "PROD_051": "Sauce / Dressing",
            "PROD_011": "Fresh Produce / Aromatics", "PROD_039": "Fresh Produce / Aromatics", "PROD_053": "Fresh Produce / Aromatics", "PROD_054": "Fresh Produce / Aromatics", "PROD_055": "Fresh Produce / Aromatics", "PROD_056": "Fresh Produce / Aromatics", "PROD_057": "Fresh Produce / Aromatics", "PROD_058": "Fresh Produce / Aromatics", "PROD_059": "Fresh Produce / Aromatics", "PROD_060": "Fresh Produce / Aromatics",
            "PROD_007": "Protein / Dairy", "PROD_008": "Protein / Dairy", "PROD_015": "Protein / Dairy", "PROD_021": "Protein / Dairy", "PROD_023": "Protein / Dairy", "PROD_024": "Protein / Dairy", "PROD_026": "Protein / Dairy", "PROD_027": "Protein / Dairy", "PROD_028": "Protein / Dairy", "PROD_040": "Protein / Dairy", "PROD_042": "Protein / Dairy", "PROD_043": "Protein / Dairy", "PROD_050": "Protein / Dairy"
        }

        all_roles = ["Base Carb", "Sauce / Dressing", "Fresh Produce / Aromatics", "Protein / Dairy"]
        found_roles = set()
        for pid in cart_product_ids:
            if pid in role_map:
                found_roles.add(role_map[pid])

        missing = [r for r in all_roles if r not in found_roles]
        score = int((len(found_roles) / len(all_roles)) * 100)

        if score >= 100:
            label = "⭐⭐⭐⭐ 100% Gastronomic Masterpiece"
            nudge = "Outstanding! Your basket contains a complete, balanced culinary meal kit ready for the kitchen."
        elif score == 75:
            label = "⭐⭐⭐ 75% Balanced Meal Kit"
            nudge = f"Nearly complete! Your cart is missing {missing[0]}. Add one to finish your dinner kit."
        elif score == 50:
            label = "⭐⭐ 50% Foundation Set"
            nudge = f"Great start! We recommend adding {missing[0]} and {missing[1]} for flavor harmony."
        else:
            label = "⭐ 25% Single Course"
            nudge = f"Quick essential trip. Add {missing[0]} and {missing[1]} to turn this into a full meal!"

        return {
            "score": score,
            "label": label,
            "found_roles": list(found_roles),
            "missing_roles": missing,
            "nudge": nudge
        }

    def generate_promotional_bundle(self, seed_product_id, bundle_size=4, discount_percent=15):
        """
        Creates a high-converting, gastronomically diverse promotional meal or pantry bundle.
        Calculates bundle price economics, customer savings, and culinary cooking narrative.
        """
        seed_info = self.miner.products_lookup.get(seed_product_id)
        if not seed_info:
            return None

        # Get diverse recommendations across aisles
        recs = self.recommend_for_product(seed_product_id, top_n=15)
        bundle_items = [
            {
                "product_id": seed_product_id,
                "product_name": seed_info["name"],
                "category": seed_info["category"],
                "price": float(seed_info["price"])
            }
        ]

        used_cats = {seed_info["category"]}
        # Prioritize items from complementary categories to ensure cross-aisle diversity
        for r in recs:
            if len(bundle_items) >= bundle_size:
                break
            bundle_items.append({
                "product_id": r["product_id"],
                "product_name": r["product_name"],
                "category": r["category"],
                "price": float(r["price"])
            })

        regular_price = sum(item["price"] for item in bundle_items)
        savings = regular_price * (discount_percent / 100.0)
        bundle_price = regular_price - savings

        # Determine bundle theme name
        category = seed_info["category"]
        if "Pasta" in category or "Italian" in category:
            bundle_name = "Authentic Italian Trattoria Dinner Bundle"
            tagline = "Slow-simmered marinara, artisan pasta & aged parmigiano"
        elif "Breakfast" in category or "Bakery" in category:
            bundle_name = "Sunday Gourmet Pancake & Breakfast Pack"
            tagline = "Buttermilk flapjacks, European sweet butter & pure Vermont syrup"
        elif "Mexican" in category:
            bundle_name = "Fiesta Street Taco & Nacho Kit"
            tagline = "Warm corn tortillas, fresh salsa, avocado guacamole & chips"
        elif "Asian" in category:
            bundle_name = "Tokyo Umami Ramen & Stir-Fry Kit"
            tagline = "Bouncy ramen noodles, fragrant sesame oil & tamari soy sauce"
        elif "Coffee" in category or "Tea" in category:
            bundle_name = "Barista Specialty Coffee Corner Pack"
            tagline = "Single-origin roast, creamy oat milk & Madagascar vanilla"
        elif "Produce" in category:
            bundle_name = "Farm-Fresh Mediterranean Salad Bundle"
            tagline = "Crisp romaine hearts, vine cherry tomatoes & cold-pressed oil"
        elif "Snacks" in category:
            bundle_name = "Ultimate Movie Night Snack Caddy"
            tagline = "Buttery microwave popcorn, kettle chips & craft ginger soda"
        else:
            bundle_name = f"Premium {seed_info['name']} Complete Essentials Bundle"
            tagline = "Hand-selected complementary kitchen favorites"

        return {
            "bundle_name": bundle_name,
            "seed_product": seed_info["name"],
            "tagline": tagline,
            "items": bundle_items,
            "regular_total_price": round(regular_price, 2),
            "bundle_promotional_price": round(bundle_price, 2),
            "discount_percentage": discount_percent,
            "customer_savings": round(savings, 2),
            "marketing_hook": f"Save {discount_percent}% with AI Chef Combo Pricing!"
        }

    # Known direct substitute clusters in quick-commerce
    SUBSTITUTE_CLUSTERS = [
        {"PROD_001", "PROD_002", "PROD_003"},  # Dry pastas (Penne, Spaghetti, Fettuccine)
        {"PROD_035", "PROD_036"},              # Tortillas (Corn, Flour)
        {"PROD_024", "PROD_025"},              # Milk (Whole Organic, Oat Milk)
        {"PROD_029", "PROD_030"},              # Coffee (Whole Bean Espresso, Ground Colombian)
        {"PROD_045", "PROD_046"},              # Grains/Noodles (Jasmine Rice, Ramen)
    ]

    def recommend_for_cart(self, cart_product_ids, top_n=5, filter_substitutes=True):
        """
        Recommends next best items for an active multi-item cart.
        Features:
        1. Multi-Item Antecedent Rule Querying ({Item A, Item B} -> Item C)
        2. Substitute Suppression Filter (Down-ranks cannibalistic duplicate staples)
        3. Cross-Category Complementarity Boost (Rewards authentic meal completion)
        4. Explainable Multi-Item Rationales
        """
        if not self.is_fitted:
            self.fit()

        if not cart_product_ids:
            return []

        cart_set = set(cart_product_ids)
        all_candidates = {}

        # 1. Multi-item and single-item cart subset association rules
        cart_rules = self.miner.get_rules_for_cart_subsets(cart_product_ids, top_n=40)
        cart_rule_dict = {}
        for r in cart_rules:
            cid = r['consequent_id']
            if cid not in cart_rule_dict or r['lift'] > cart_rule_dict[cid]['lift']:
                cart_rule_dict[cid] = r

        # 2. Collect item-level recommendations and combine with multi-item rules
        for pid in cart_product_ids:
            recs = self.recommend_for_product(pid, top_n=15)
            for r in recs:
                target_id = r["product_id"]
                if target_id in cart_set:
                    continue
                if target_id not in all_candidates:
                    all_candidates[target_id] = r.copy()
                    all_candidates[target_id]["cart_hits"] = 1
                else:
                    all_candidates[target_id]["hybrid_score"] += (r["hybrid_score"] * 0.7)
                    all_candidates[target_id]["cart_hits"] += 1
                    if r["transaction_lift"] > all_candidates[target_id]["transaction_lift"]:
                        all_candidates[target_id]["transaction_lift"] = r["transaction_lift"]

        # Incorporate multi-item rule candidates if not already present
        for cid, rule_info in cart_rule_dict.items():
            if cid in cart_set:
                continue
            if cid in all_candidates:
                ant_len = rule_info.get('antecedent_len', 1)
                bonus = 0.25 * ant_len * min(rule_info['lift'] / 5.0, 1.5)
                all_candidates[cid]["hybrid_score"] += bonus
                all_candidates[cid]["transaction_lift"] = max(all_candidates[cid]["transaction_lift"], rule_info['lift'])
                all_candidates[cid]["multi_item_rule"] = rule_info
            else:
                prod_info = self.miner.products_lookup.get(cid, {})
                sent_info = self.nlp.product_sentiment.get(cid, {})
                all_candidates[cid] = {
                    "product_id": cid,
                    "product_name": prod_info.get('name', cid),
                    "category": prod_info.get('category', 'Unknown'),
                    "price": float(prod_info.get('price', 0.0)),
                    "hybrid_score": float(rule_info['lift'] / 5.0),
                    "transaction_lift": float(rule_info['lift']),
                    "transaction_confidence": float(rule_info['confidence']),
                    "transaction_support": float(rule_info['support']),
                    "nlp_semantic_similarity": 0.25,
                    "customer_sentiment": sent_info.get('sentiment_label', 'Positive'),
                    "customer_rating": sent_info.get('mean_rating', 4.5),
                    "cart_hits": 1,
                    "multi_item_rule": rule_info,
                    "explanation": f"Shoppers frequently buy this together with your cart items ({rule_info['lift']:.1f}x Co-purchase Lift)"
                }

        # 3. Apply Substitute Suppression & Complementary Boost (Retail Economics)
        if filter_substitutes:
            represented_clusters = []
            for cluster in self.SUBSTITUTE_CLUSTERS:
                if cluster.intersection(cart_set):
                    represented_clusters.append(cluster)

            cart_categories = {self.miner.products_lookup.get(pid, {}).get('category') for pid in cart_set}

            for cid, c_data in all_candidates.items():
                is_substitute = any(cid in cluster for cluster in represented_clusters)
                if is_substitute and cid not in cart_set:
                    c_data["hybrid_score"] -= 0.60
                    c_data["is_substitute_penalized"] = True

                cand_cat = c_data.get('category')
                if cand_cat and cand_cat not in cart_categories:
                    c_data["hybrid_score"] += 0.20
                    c_data["is_cross_aisle_complement"] = True

        # 4. Refine human-readable explanation with multi-item context
        for cid, c_data in all_candidates.items():
            if "multi_item_rule" in c_data and c_data["multi_item_rule"].get('antecedent_len', 1) > 1:
                ant_names = c_data["multi_item_rule"]['antecedent_names']
                c_data["explanation"] = f"🔥 {c_data['transaction_lift']:.1f}x Co-purchase combo: Customers with {ant_names} heavily add this item (Confidence: {c_data['multi_item_rule']['confidence']*100:.1f}%)"

        ranked = sorted(all_candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_n]

if __name__ == "__main__":
    recommender = HybridRecommender()
    recommender.fit()
    
    print("\n=======================================================")
    print("DEMO: Hybrid Recommendations for 'Italian Penne Rigate Pasta' (PROD_001)")
    print("=======================================================")
    recs = recommender.recommend_for_product("PROD_001", top_n=4)
    for i, r in enumerate(recs, 1):
        print(f"\n{i}. {r['product_name']} (${r['price']:.2f})")
        print(f"   Hybrid Score: {r['hybrid_score']} | Lift: {r['transaction_lift']}x | Confidence: {r['transaction_confidence']*100:.1f}%")
        print(f"   NLP Similarity: {r['nlp_semantic_similarity']} | Rating: {r['customer_rating']}★")
        print(f"   Why: {r['explanation']}")

    print("\n=======================================================")
    print("DEMO: Smart Promotional Bundle")
    print("=======================================================")
    bundle = recommender.generate_promotional_bundle("PROD_001", bundle_size=4, discount_percent=15)
    print(f"Bundle: {bundle['bundle_name']}")
    for itm in bundle['items']:
        print(f" - {itm['product_name']}: ${itm['price']:.2f}")
    print(f"Regular Price: ${bundle['regular_total_price']:.2f} -> Bundle Price: ${bundle['bundle_promotional_price']:.2f} (Save ${bundle['customer_savings']:.2f} - {bundle['discount_percentage']}%)")
