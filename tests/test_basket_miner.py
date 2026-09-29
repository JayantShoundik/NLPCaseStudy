"""
test_basket_miner.py - Unit tests for Apriori and FP-Growth association rule mining.
Includes deterministic toy verification and algorithm equivalence assertions.
"""

import os
import sys
import unittest
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.market_basket_miner import MarketBasketMiner

class TestBasketMiner(unittest.TestCase):
    def test_deterministic_toy_dataset_metrics(self):
        """
        Hand-crafted 4-transaction toy dataset:
        T1: ['apple', 'beer', 'rice']
        T2: ['apple', 'beer']
        T3: ['apple', 'rice']
        T4: ['beer', 'rice']
        
        Support(apple, beer) = 2/4 = 0.50
        Support(apple) = 3/4 = 0.75
        Support(beer) = 3/4 = 0.75
        Confidence(apple -> beer) = 0.50 / 0.75 = 0.6667
        Lift(apple -> beer) = 0.6667 / 0.75 = 0.8889
        """
        toy_transactions = [
            ['apple', 'beer', 'rice'],
            ['apple', 'beer'],
            ['apple', 'rice'],
            ['beer', 'rice']
        ]
        te = TransactionEncoder()
        df_encoded = pd.DataFrame(te.fit(toy_transactions).transform(toy_transactions), columns=te.columns_)
        
        # Mine with Apriori
        freq_ap = apriori(df_encoded, min_support=0.4, use_colnames=True)
        rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=0.5)

        # Check apple -> beer
        matched = rules_ap[(rules_ap['antecedents'] == frozenset(['apple'])) & 
                           (rules_ap['consequents'] == frozenset(['beer']))]
        self.assertEqual(len(matched), 1)
        r = matched.iloc[0]
        self.assertAlmostEqual(r['support'], 0.50, places=2)
        self.assertAlmostEqual(r['confidence'], 2.0 / 3.0, places=3)
        self.assertAlmostEqual(r['lift'], (2.0 / 3.0) / 0.75, places=3)

    def test_apriori_fpgrowth_equivalence(self):
        """
        Verifies that Apriori and FP-Growth output the EXACT same frequent itemsets
        given identical minimum support threshold.
        """
        miner = MarketBasketMiner()
        miner.load_data(train_only=True)
        miner.encode_transactions()
        
        # Test on 2,000 transactions subset for quick validation
        sub = miner.df_encoded.iloc[:2000]
        
        fp_sets = fpgrowth(sub, min_support=0.03, use_colnames=True)
        ap_sets = apriori(sub, min_support=0.03, use_colnames=True)
        
        # Both must find identical number of itemsets
        self.assertEqual(len(fp_sets), len(ap_sets), "FP-Growth and Apriori must find equal number of itemsets")
        
        # Sort itemsets into comparable string keys
        fp_keys = set(fp_sets['itemsets'].apply(lambda s: tuple(sorted(list(s)))))
        ap_keys = set(ap_sets['itemsets'].apply(lambda s: tuple(sorted(list(s)))))
        
        self.assertEqual(fp_keys, ap_keys, "Frequent itemset keys from Apriori and FP-Growth must be identical")

    def test_association_rules_ranges(self):
        miner = MarketBasketMiner()
        miner.fit_all(min_support=0.02, min_lift=1.2)
        rules = miner.rules_df
        
        self.assertGreater(len(rules), 100, "Should discover more than 100 rules")
        self.assertTrue((rules['support'] > 0).all() and (rules['support'] <= 1.0).all())
        self.assertTrue((rules['confidence'] > 0).all() and (rules['confidence'] <= 1.0).all())
        self.assertTrue((rules['lift'] >= 1.2).all(), "All extracted rules should satisfy minimum lift threshold")

if __name__ == '__main__':
    unittest.main()
