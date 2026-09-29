# Project 1: Market Basket Analysis with NLP & Association Mining

An end-to-end Case Study fusing **Market Basket Analysis (Apriori & FP-Growth)** with **Natural Language Processing (NLP)** on product descriptions and customer reviews to generate data-driven, explainable product recommendations and promotional bundles.

---

## 🌟 Key Features

1. **High-Volume Real-World Scale**:
   - **50,000 Transaction Baskets** across 10 diverse quick-commerce supermarket aisles.
   - **80/20 Train/Test Split**: 40,000 training baskets vs. 10,000 unseen holdout test baskets.
   - **74 SKUs**: Detailed descriptions, ingredients, pricing, pack specs, and 100% verified distinct CDN photos.
   - **15,000 Customer Reviews**: Verified shopper reviews with VADER sentiment scores and culinary co-mentions.

2. **🍳 AI Recipe & Meal Kit Copilot (Natural Language Prompt-to-Cart Engine)**:
   - Free-form natural language query processing (e.g., *"Authentic creamy Italian pasta dinner for date night"*, *"Quick Mexican street tacos under $15"*).
   - NLP Intent & Entity Extraction: Identifies cuisine themes, occasion, and dietary tags.
   - Semantic Vector Ranking & Culinary Blueprint: Selects `[Base Carb]` + `[Sauce/Marinade]` + `[Fresh Produce/Aromatic]` + `[Protein/Dairy]` + `[Side/Beverage]`.
   - Generates **AI Culinary Gastronomy Narrative**, 4-step chef recipe instructions, and **1-Click Load into Cart**.

3. **🗺️ 2D Semantic Vector Embedding Map (TruncatedSVD / PCA)**:
   - Projects high-dimensional 2,500-term TF-IDF vector matrix into 2D coordinates across all 74 products.
   - Interactive Plotly scatter plot showing natural clustering of products by culinary proximity.
   - Highlights items currently in the cart with gold stars and connecting lines.
   - Computes live **Cosine Similarity Heatmap** between items in the active cart.

4. **💬 Aspect-Based Sentiment Analysis (ABSA)**:
   - Evaluates customer reviews across 4 core dimensions: 👅 Taste & Flavor, 🌿 Freshness & Quality, 📦 Texture & Consistency, 💰 Value for Money.
   - Computes percentage positive sentiment gauges, extracted food pairings, and sentiment quotes.

5. **🧪 Zero-History Cold-Start Problem Solver**:
   - Demonstrates how NLP solves the classic e-commerce Cold Start failure: When a brand-new product has 0 receipts (Apriori/FP-Growth produces 0 rules), our NLP Semantic Engine vectorizes description & ingredients to immediately generate 5 complementary cross-sell recommendations.

6. **⚡ Real Quick-Commerce Business Logic**:
   - **Dynamic 10-Minute Delivery Economics**: Live progress bar towards FREE delivery at $15.00 threshold ($1.99 standard fee below threshold).
   - **❄️ Cold-Chain Packaging Compliance**: Automatically identifies dairy, eggs, and fresh produce to include insulated thermal ice pouch packaging.
   - **🥘 AI Basket Gastronomic Health Diagnostic**: Evaluates active basket against culinary balance and provides real-time intelligent culinary nudges on missing meal components.

7. **Algorithmic Association Rule Mining**:
   - **Apriori Algorithm**: Level-wise candidate generation and pruning.
   - **FP-Growth Algorithm**: High-speed divide-and-conquer frequent pattern tree mining (demonstrating **up to 15.9x speedup** over Apriori).
   - Metrics: `Support`, `Confidence`, `Lift`, `Leverage`, `Conviction`.

8. **Rigorous Offline Holdout Testing**:
   - Evaluates on 1,000 unseen shopping sessions measuring **Hit Rate@K**, **Precision@K**, **Recall@K**, and **Mean Reciprocal Rank (MRR)**.
   - The Hybrid model achieves an **81.3% Hit Rate@5**, substantially outperforming popularity baselines (39.6%) and pure NLP (67.9%).

---

## 📁 Project Structure

```
NLPCaseStudy/
├── data/
│   ├── products.csv                 # 74 products with descriptions & ingredients
│   ├── transactions.csv             # 50,000 total transaction baskets
│   ├── train_transactions.csv       # 40,000 training baskets (80%)
│   ├── test_transactions.csv        # 10,000 holdout test baskets (20%)
│   └── reviews.csv                  # 15,000 customer reviews with ratings
├── src/
│   ├── data_pipeline.py             # High-throughput realistic data generation
│   ├── download_public_datasets.py  # Utility for external benchmarks (UCI Retail)
│   ├── nlp_engine.py                # TF-IDF, cosine similarity, VADER sentiment
│   ├── market_basket_miner.py       # Apriori & FP-Growth algorithms and profiling
│   ├── hybrid_recommender.py        # Fusion model, bundles, & explainable AI
│   ├── evaluate_recommender.py      # Holdout test suite (Hit Rate, Precision, MRR)
│   └── dashboard.py                 # Streamlit interactive analytics web app
├── tests/
│   ├── test_data_pipeline.py        # Unit tests for schema & distributions
│   ├── test_nlp_engine.py           # Unit tests for text cleaning & embeddings
│   ├── test_basket_miner.py         # Unit tests for toy deterministic math & equivalence
│   └── test_hybrid_recommender.py   # Unit tests for ranking & bundle economics
├── reports/
│   ├── rule_scatter_plot.png        # Support vs Confidence colored by Lift
│   ├── apriori_vs_fpgrowth_benchmark.png # Runtime scalability curves
│   ├── product_similarity_heatmap.png   # NLP semantic correlation matrix
│   ├── model_evaluation_comparison.png  # Hit Rate comparison across models
│   ├── model_evaluation_metrics.json    # Quantitative test metrics
│   └── top_promotional_bundles.json     # Generated promotional packages
├── notebooks/
│   └── case_study_walkthrough.ipynb # Comprehensive academic / portfolio notebook
├── requirements.txt
├── run_case_study.py                # Master headless pipeline orchestrator
└── README.md
```

---

## 🚀 How to Run

### 1. Run the Complete End-to-End Case Study
Executes data verification, NLP processing, association mining, bundle creation, holdout test evaluation, and chart generation:
```bash
python3 run_case_study.py
```

### 2. Launch the Interactive Streamlit Dashboard
```bash
python3 -m streamlit run src/dashboard.py
```
Open your browser at `http://localhost:8501` to explore association rules, benchmark curves, NLP embeddings, and the live shopping cart simulator.

### 3. Run the Automated Unit Test Suite
```bash
python3 -m unittest discover -s tests -v
```

---

## 📊 Evaluation Summary

| Model | Hit Rate@3 | Hit Rate@5 | Precision@5 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 31.7% | 39.6% | 15.7% | 16.1% | 0.2661 |
| **Pure NLP Content-Based** | 56.4% | 67.9% | 24.2% | 26.2% | 0.5200 |
| **Pure Association Mining** | 71.4% | 80.3% | 45.0% | 50.5% | 0.6205 |
| **Hybrid (Association + NLP + Reviews)** | **72.7%** | **81.3%** | **42.2%** | **46.8%** | **0.6453** |

*Evaluated on 1,000 unseen test shopping sessions from the holdout test set.*
