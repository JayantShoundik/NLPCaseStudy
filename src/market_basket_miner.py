"""
market_basket_miner.py - Market Basket Analysis using Apriori and FP-Growth
Encodes transactions, mines frequent itemsets, computes association metrics
(Support, Confidence, Lift, Leverage, Conviction), and benchmarks algorithm efficiency.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

class MarketBasketMiner:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.transactions_list = []
        self.encoder = None
        self.df_encoded = None
        self.products_lookup = {}
        self.frequent_itemsets = None
        self.rules_df = None
        self.benchmark_results = []

    def load_data(self, train_only=True):
        """Loads transaction data and product metadata."""
        file_name = "train_transactions.csv" if train_only else "transactions.csv"
        tx_path = os.path.join(self.data_dir, file_name)
        prod_path = os.path.join(self.data_dir, "products.csv")

        if not os.path.exists(tx_path) or not os.path.exists(prod_path):
            raise FileNotFoundError(f"Data files not found in {self.data_dir}. Run data_pipeline.py first.")

        df_tx = pd.read_csv(tx_path)
        df_prod = pd.read_csv(prod_path)
        
        self.products_lookup = df_prod.set_index('id').to_dict('index')
        self.transactions_list = [str(items).split(',') for items in df_tx['items']]
        print(f"Loaded {len(self.transactions_list):,} transactions for basket mining.")

    def encode_transactions(self):
        """One-hot encodes transactions into a boolean matrix using TransactionEncoder."""
        print("One-hot encoding transactions into sparse matrix...")
        t0 = time.time()
        self.encoder = TransactionEncoder()
        te_ary = self.encoder.fit(self.transactions_list).transform(self.transactions_list)
        self.df_encoded = pd.DataFrame(te_ary, columns=self.encoder.columns_)
        print(f"Encoded matrix shape: {self.df_encoded.shape} in {time.time() - t0:.2f}s")
        return self.df_encoded

    def run_fpgrowth(self, min_support=0.01, use_colnames=True):
        """Runs the FP-Growth algorithm to find frequent itemsets."""
        print(f"Running FP-Growth with min_support={min_support}...")
        t0 = time.time()
        itemsets = fpgrowth(self.df_encoded, min_support=min_support, use_colnames=use_colnames)
        elapsed = time.time() - t0
        itemsets['itemset_length'] = itemsets['itemsets'].apply(lambda x: len(x))
        print(f"FP-Growth found {len(itemsets):,} frequent itemsets in {elapsed:.3f}s")
        self.frequent_itemsets = itemsets
        return itemsets, elapsed

    def run_apriori(self, min_support=0.01, use_colnames=True):
        """Runs the Apriori algorithm to find frequent itemsets."""
        print(f"Running Apriori with min_support={min_support}...")
        t0 = time.time()
        itemsets = apriori(self.df_encoded, min_support=min_support, use_colnames=use_colnames)
        elapsed = time.time() - t0
        itemsets['itemset_length'] = itemsets['itemsets'].apply(lambda x: len(x))
        print(f"Apriori found {len(itemsets):,} frequent itemsets in {elapsed:.3f}s")
        return itemsets, elapsed

    def benchmark_algorithms(self, sample_size=15000, support_thresholds=[0.05, 0.03, 0.02, 0.015, 0.01]):
        """
        Benchmarks execution time of Apriori vs FP-Growth across multiple support thresholds.
        Uses a representative sample for fast, direct comparison.
        """
        print(f"\n--- Benchmarking Apriori vs FP-Growth (Sample: {sample_size:,} baskets) ---")
        sub_encoded = self.df_encoded.iloc[:sample_size]
        results = []

        for sup in support_thresholds:
            print(f"Testing min_support = {sup:.3f}...")
            
            # FP-Growth
            t0 = time.time()
            fp_items = fpgrowth(sub_encoded, min_support=sup, use_colnames=True)
            fp_time = time.time() - t0
            
            # Apriori
            t0 = time.time()
            ap_items = apriori(sub_encoded, min_support=sup, use_colnames=True)
            ap_time = time.time() - t0

            speedup = ap_time / fp_time if fp_time > 0 else 1.0

            results.append({
                "min_support": sup,
                "itemsets_found": len(fp_items),
                "fpgrowth_time_sec": round(fp_time, 4),
                "apriori_time_sec": round(ap_time, 4),
                "speedup_factor": round(speedup, 2)
            })

        self.benchmark_results = results
        df_bench = pd.DataFrame(results)
        print("\nBenchmark Summary:")
        print(df_bench.to_string(index=False))
        return df_bench

    def mine_association_rules(self, metric="lift", min_threshold=1.2):
        """
        Extracts association rules from frequent itemsets and enriches them
        with human-readable product names and category metadata.
        """
        if self.frequent_itemsets is None or len(self.frequent_itemsets) == 0:
            self.run_fpgrowth(min_support=0.01)

        print(f"Mining association rules (metric='{metric}', min_threshold={min_threshold})...")
        rules = association_rules(self.frequent_itemsets, metric=metric, min_threshold=min_threshold)

        # Convert frozensets to lists and string representations
        rules['antecedent_ids'] = rules['antecedents'].apply(lambda s: sorted(list(s)))
        rules['consequent_ids'] = rules['consequents'].apply(lambda s: sorted(list(s)))
        
        # Human readable names
        def ids_to_names(id_list):
            return ", ".join([self.products_lookup.get(pid, {}).get('name', pid) for pid in id_list])

        rules['antecedent_names'] = rules['antecedent_ids'].apply(ids_to_names)
        rules['consequent_names'] = rules['consequent_ids'].apply(ids_to_names)
        rules['antecedent_len'] = rules['antecedent_ids'].apply(len)
        rules['consequent_len'] = rules['consequent_ids'].apply(len)

        # Round metrics
        for col in ['support', 'confidence', 'lift', 'leverage', 'conviction']:
            if col in rules.columns:
                rules[col] = rules[col].apply(lambda x: round(float(x), 4) if np.isfinite(x) else 999.99)

        # Sort by lift descending
        rules = rules.sort_values(by='lift', ascending=False).reset_index(drop=True)
        self.rules_df = rules
        print(f"Extracted {len(rules):,} association rules above Lift {min_threshold}.")
        return rules

    def get_rules_for_product(self, product_id, top_n=10):
        """Returns top association rules where the product is the sole antecedent."""
        if self.rules_df is None:
            self.mine_association_rules()

        # Match rules where product_id is in antecedent
        matched = self.rules_df[
            (self.rules_df['antecedent_ids'].apply(lambda x: [product_id] == x)) &
            (self.rules_df['consequent_len'] == 1)
        ].head(top_n)

        results = []
        for _, row in matched.iterrows():
            target_id = row['consequent_ids'][0]
            prod_info = self.products_lookup.get(target_id, {})
            results.append({
                "consequent_id": target_id,
                "consequent_name": prod_info.get('name', target_id),
                "category": prod_info.get('category', 'Unknown'),
                "support": row['support'],
                "confidence": row['confidence'],
                "lift": row['lift'],
                "price": prod_info.get('price', 0.0)
            })
        return results

    def get_rules_for_cart_subsets(self, cart_ids, top_n=25):
        """
        Finds association rules where the antecedent is a subset of the active cart
        and the consequent is NOT in the active cart.
        Supports multi-item antecedents (e.g. {Penne, Marinara} -> Parmigiano).
        """
        if self.rules_df is None:
            self.mine_association_rules()

        cart_set = set(cart_ids)
        if not cart_set:
            return []

        # Find rules where antecedent is subset of cart_set and consequent is NOT in cart_set
        def is_valid_cart_rule(row):
            ant_set = set(row['antecedent_ids'])
            cons_set = set(row['consequent_ids'])
            return ant_set.issubset(cart_set) and not cons_set.intersection(cart_set) and len(cons_set) == 1

        matched = self.rules_df[self.rules_df.apply(is_valid_cart_rule, axis=1)]
        
        # Sort by antecedent length descending (multi-item rules prioritized), then lift descending
        matched = matched.sort_values(by=['antecedent_len', 'lift', 'confidence'], ascending=[False, False, False]).head(top_n)

        results = []
        for _, row in matched.iterrows():
            target_id = row['consequent_ids'][0]
            prod_info = self.products_lookup.get(target_id, {})
            results.append({
                "consequent_id": target_id,
                "consequent_name": prod_info.get('name', target_id),
                "category": prod_info.get('category', 'Unknown'),
                "antecedent_ids": row['antecedent_ids'],
                "antecedent_names": row['antecedent_names'],
                "antecedent_len": row['antecedent_len'],
                "support": row['support'],
                "confidence": row['confidence'],
                "lift": row['lift'],
                "price": prod_info.get('price', 0.0)
            })
        return results

    def fit_all(self, min_support=0.015, min_lift=1.2):
        """Runs complete mining pipeline."""
        self.load_data(train_only=True)
        self.encode_transactions()
        self.run_fpgrowth(min_support=min_support)
        self.mine_association_rules(metric="lift", min_threshold=min_lift)
        return self

if __name__ == "__main__":
    miner = MarketBasketMiner()
    miner.fit_all(min_support=0.015, min_lift=1.2)
    
    print("\n--- Top 5 Association Rules Overall ---")
    cols = ['antecedent_names', 'consequent_names', 'support', 'confidence', 'lift']
    print(miner.rules_df[cols].head(5).to_string(index=False))

    print("\n--- Direct Complements for 'Italian Penne Rigate Pasta' (PROD_001) ---")
    rules_pasta = miner.get_rules_for_product("PROD_001", top_n=5)
    for r in rules_pasta:
        print(f" -> {r['consequent_name']} | Lift: {r['lift']:.2f} | Confidence: {r['confidence']*100:.1f}% | Support: {r['support']*100:.2f}%")
