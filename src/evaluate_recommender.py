"""
evaluate_recommender.py - Rigorous Holdout Test Evaluation of Market Basket Recommenders
Evaluates models on unseen test transactions (10,000 baskets) measuring Hit Rate@K,
Precision@K, Recall@K, and MRR across Popularity, Pure Association, Pure NLP, and Hybrid models.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hybrid_recommender import HybridRecommender

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

class ModelEvaluator:
    def __init__(self, data_dir=DATA_DIR, reports_dir=REPORTS_DIR):
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        
        self.recommender = HybridRecommender(data_dir=data_dir)
        self.test_baskets = []
        self.popular_items = []

    def setup(self):
        """Loads models and test transactions."""
        print("Training recommendation models on 40,000 training transactions...")
        self.recommender.fit(min_support=0.015, min_lift=1.2)
        
        # Determine popularity baseline from train set
        all_train_items = [item for tx in self.recommender.miner.transactions_list for item in tx]
        item_counts = Counter(all_train_items)
        self.popular_items = [item for item, _ in item_counts.most_common(20)]

        # Load unseen test transactions
        test_path = os.path.join(self.data_dir, "test_transactions.csv")
        df_test = pd.read_csv(test_path)
        self.test_baskets = [items.split(',') for items in df_test['items'] if len(items.split(',')) >= 3]
        print(f"Loaded {len(self.test_baskets):,} multi-item test transactions for holdout evaluation.")

    def predict_popularity(self, given_items, k=5):
        """Popularity baseline prediction."""
        preds = [item for item in self.popular_items if item not in given_items]
        return preds[:k]

    def predict_association_only(self, given_items, k=5):
        """Pure Association Rule prediction (based on Lift)."""
        rules_df = self.recommender.miner.rules_df
        candidates = {}
        for item in given_items:
            sub = rules_df[
                (rules_df['antecedent_ids'].apply(lambda x: [item] == x)) &
                (rules_df['consequent_len'] == 1)
            ]
            for _, r in sub.iterrows():
                target = r['consequent_ids'][0]
                if target not in given_items:
                    candidates[target] = max(candidates.get(target, 0.0), r['lift'])
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:k]]

    def predict_nlp_only(self, given_items, k=5):
        """Pure NLP Content-based similarity prediction."""
        candidates = {}
        for item in given_items:
            sims = self.recommender.nlp.get_top_nlp_similar_products(item, top_n=10)
            for s in sims:
                target = s['product_id']
                if target not in given_items:
                    candidates[target] = max(candidates.get(target, 0.0), s['similarity_score'])
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in ranked[:k]]

    def predict_hybrid(self, given_items, k=5):
        """Hybrid Association + NLP + Review Sentiment prediction."""
        recs = self.recommender.recommend_for_cart(given_items, top_n=k)
        return [r['product_id'] for r in recs]

    def evaluate_all_models(self, test_sample_size=1000, k_values=[3, 5]):
        """
        Runs rigorous holdout evaluation across test samples.
        Computes Hit Rate@K, Precision@K, Recall@K, and MRR.
        """
        sample_baskets = self.test_baskets[:test_sample_size]
        print(f"\nEvaluating models on {len(sample_baskets):,} unseen test shopping sessions...")

        models = {
            "Popularity Baseline": self.predict_popularity,
            "Pure Association Rules (Apriori/FP-Growth)": self.predict_association_only,
            "Pure NLP Semantic Similarity": self.predict_nlp_only,
            "Hybrid (Association + NLP + Reviews)": self.predict_hybrid
        }

        results = []

        for model_name, predict_func in models.items():
            print(f"Evaluating {model_name}...")
            t0 = time.time()
            metrics = {k: {"hits": 0, "precisions": [], "recalls": [], "mrrs": []} for k in k_values}

            for basket in sample_baskets:
                # Seed item: first item, Ground truth: remaining items
                seed_items = [basket[0]]
                ground_truth = set(basket[1:])

                for k in k_values:
                    preds = predict_func(seed_items, k=k)
                    pred_set = set(preds)
                    hits = len(pred_set.intersection(ground_truth))

                    # Hit rate
                    if hits > 0:
                        metrics[k]["hits"] += 1

                    # Precision & Recall
                    precision = hits / k if k > 0 else 0.0
                    recall = hits / len(ground_truth) if len(ground_truth) > 0 else 0.0
                    metrics[k]["precisions"].append(precision)
                    metrics[k]["recalls"].append(recall)

                    # Reciprocal Rank
                    rr = 0.0
                    for rank, pred_item in enumerate(preds, 1):
                        if pred_item in ground_truth:
                            rr = 1.0 / rank
                            break
                    metrics[k]["mrrs"].append(rr)

            for k in k_values:
                hit_rate = metrics[k]["hits"] / len(sample_baskets)
                avg_prec = np.mean(metrics[k]["precisions"])
                avg_rec = np.mean(metrics[k]["recalls"])
                avg_mrr = np.mean(metrics[k]["mrrs"])

                results.append({
                    "Model": model_name,
                    "K": k,
                    "Hit Rate@K": round(hit_rate, 4),
                    "Precision@K": round(avg_prec, 4),
                    "Recall@K": round(avg_rec, 4),
                    "MRR": round(avg_mrr, 4),
                    "Eval Time (s)": round(time.time() - t0, 2)
                })

        df_results = pd.DataFrame(results)
        print("\n=======================================================")
        print("OUT-OF-SAMPLE TEST EVALUATION SUMMARY")
        print("=======================================================")
        print(df_results.to_string(index=False))

        # Save to reports
        csv_path = os.path.join(self.reports_dir, "model_evaluation_metrics.csv")
        json_path = os.path.join(self.reports_dir, "model_evaluation_metrics.json")
        df_results.to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved evaluation metrics to:\n - {csv_path}\n - {json_path}")
        return df_results

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.setup()
    evaluator.evaluate_all_models(test_sample_size=1000, k_values=[3, 5])
