"""
test_nlp_engine.py - Unit tests for text preprocessing, TF-IDF semantic similarities,
VADER sentiment analysis, and aspect/pairing keyword extraction.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.nlp_engine import NLPEngine, clean_and_lemmatize

class TestNLPEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = NLPEngine()
        cls.engine.fit_all()

    def test_clean_and_lemmatize(self):
        sample = "The organic tomatoes were freshly cooked into delicious sauces!"
        cleaned = clean_and_lemmatize(sample)
        # Check that stopwords like 'the', 'into', 'were' are stripped and words are lowercased
        self.assertNotIn("the", cleaned.split())
        self.assertNotIn("were", cleaned.split())
        self.assertIn("tomato", cleaned.split()) # lemmatized

    def test_semantic_similarity_relative_ordering(self):
        # Pasta (PROD_001) should be more semantically similar to Marinara Sauce (PROD_004)
        # than to Dish Soap (PROD_071)
        sim_pasta_sauce = self.engine.get_semantic_similarity("PROD_001", "PROD_004")
        sim_pasta_soap = self.engine.get_semantic_similarity("PROD_001", "PROD_071")
        
        self.assertGreater(sim_pasta_sauce, sim_pasta_soap,
                           f"Expected sim(Pasta, Sauce)={sim_pasta_sauce:.3f} > sim(Pasta, Soap)={sim_pasta_soap:.3f}")

    def test_vader_sentiment_distribution(self):
        sentiments = self.engine.product_sentiment
        self.assertGreaterEqual(len(sentiments), 60)
        # Check that average sentiment scores are bounded in [-1.0, 1.0]
        for pid, stats in sentiments.items():
            self.assertGreaterEqual(stats['mean_sentiment'], -1.0)
            self.assertLessEqual(stats['mean_sentiment'], 1.0)
            self.assertIn(stats['sentiment_label'], ["Highly Positive", "Positive", "Neutral", "Negative"])

    def test_pairing_insights_extraction(self):
        insights = self.engine.pairing_insights
        self.assertIn("PROD_001", insights)
        penne_info = insights["PROD_001"]
        self.assertTrue(len(penne_info["sample_quote"]) > 0)

if __name__ == '__main__':
    unittest.main()
