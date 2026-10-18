# Advanced Fake News Detection & Explainable Misinformation Analysis
### Using NLP, Machine Learning, Transformer Models, Probability Calibration, and MySQL

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![SQLAlchemy 2.x](https://img.shields.io/badge/SQLAlchemy-2.x-red.svg)](https://www.sqlalchemy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

---

## 1. Project Overview & Scientific Objectives

Automated misinformation classification is an urgent area of study in Natural Language Processing (NLP). This repository provides an enterprise-grade, reproducible research and engineering pipeline that classifies news articles as resembling either the real or fake news examples in the benchmark dataset, while maintaining rigorous scientific humility:

* **Primary Deliverable:** `Advanced_Fake_News_Detection.ipynb` (Fully executable Jupyter Notebook across 22 numbered sections).
* **Baseline and Advanced Machine Learning:** DummyClassifier, Multinomial Naive Bayes, Logistic Regression, Linear SVC, Random Forest, and Gradient Boosted Trees.
* **Feature Engineering:** Sublinear Word TF-IDF, Character N-Grams, and Custom Scikit-Learn Stylometric Transformers (casing ratio, punctuation density, sentence metrics).
* **Deep Learning & Transformers:** Full PyTorch and Hugging Face Transformers pipeline (`distilbert-base-uncased`) with graceful hardware fallbacks.
* **Probability Calibration:** Reliability diagrams, Brier score optimization, and isotonic calibration to prevent overconfidence.
* **Explainable AI (XAI):** Linear model coefficient attribution, top positive/negative $n$-grams, tree feature importance, and sample-level attribution.
* **Data Leakage Prevention:** Cryptographic SHA-256 deduplication and leak-free 3-way stratified train/val/test splitting.
* **Database Engineering:** MySQL 8+ / SQLAlchemy 2.x integration for articles, predictions, experiments, model registry, and users, featuring seamless local SQLite fallback.

---

> ### ⚠️ Mandatory Scientific Limitation & Ethical Disclaimer
> **This model does NOT claim to verify objective factual truth.**
> NLP classifiers evaluate statistical similarity to labeled examples from a specific training corpus. A classification of `REAL` or `FAKE` reflects statistical correlation with the dataset annotations, **not independently verified factuality**. Real-world fact-checking requires external evidence retrieval, cross-source verification, and investigative journalism. All predictions must be interpreted as dataset-based classifications.

---

## 2. Repository Architecture & Directory Structure

```text
Fake News Detection Using NLP/
├── Advanced_Fake_News_Detection.ipynb   # PRIMARY DELIVERABLE: 22-section research notebook
├── requirements.txt                    # Production and research dependencies
├── README.md                           # Comprehensive documentation and reproducibility guide
├── .env.example                        # Template for database & pipeline environment variables
├── .gitignore                          # Exclusions for data, models, cache, and secrets
├── data/
│   ├── raw/
│   │   ├── Fake.csv                    # Kaggle Fake news dataset (23,481 records)
│   │   └── True.csv                    # Kaggle Real news dataset (21,417 records)
│   └── processed/                      # Intermediate preprocessed caches
├── artifacts/
│   └── models/
│       ├── fake_news_logistic_pipeline.joblib # Serialized best production pipeline
│       └── model_metadata.json         # Artifact version, hyperparameters & metrics
├── outputs/
│   ├── model_comparison.csv            # Comprehensive test set evaluation metrics
│   ├── classification_reports.csv      # Per-class precision, recall, and F1 reports
│   ├── misclassified_examples.csv      # Qualitative error analysis export
│   ├── data_quality_report.csv         # Audit metrics on missingness & duplicates
│   ├── experiment_summary.json         # Structured metadata summary of experiment run
│   ├── final_report.md                 # Executive research and benchmark report
│   ├── data_quality_audit.png          # Visual audit of length, label & subject distributions
│   ├── model_roc_pr_curves.png         # Comparative ROC and Precision-Recall curves
│   ├── calibration_reliability_curve.png # Reliability diagram before and after calibration
│   ├── top_indicative_features.png     # Top 20 positive and negative indicative n-grams
│   └── fake_news.db                    # Local SQLite audit database (fallback)
├── sql/
│   └── schema.sql                      # Production MySQL 8.0+ DDL migration schema
└── src/
    ├── __init__.py                     # Package initialization
    ├── data_processing.py              # Ingestion, schema audit, and feature extractors
    ├── database.py                     # SQLAlchemy 2.x ORM models & connection managers
    ├── evaluation.py                   # Metrics, calibration curves, and perturbation tests
    └── inference.py                    # Real-time prediction, batch CSV, and XAI explainers
```

---

## 3. Dataset Configuration & Acquisition

The project utilizes the **Fake and Real News Dataset** originally compiled by Clément Bisaillon:
* **Kaggle URL:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* **Expected Files:** `Fake.csv` (23,481 rows) and `True.csv` (21,417 rows).
* **Expected Columns:** `title`, `text`, `subject`, `date`.
* **Label Mapping:** `0 = Fake`, `1 = Real`.

### Workflow A: Manual Download
1. Download `Fake.csv` and `True.csv` from the Kaggle dataset link above.
2. Place both CSV files into `data/raw/` (or the project root directory).

### Workflow B: Kaggle API CLI Download
```bash
# Ensure Kaggle API token is configured in ~/.kaggle/kaggle.json
pip install kaggle
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw/ --unzip
```

---

## 4. Installation & Environment Setup

### 1. Clone & Create Virtual Environment
```bash
git clone <repository_url>
cd "Fake News Detection Using NLP"

# Create a virtual environment using Python 3.11 or compatible
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional Deep Learning / GPU Acceleration
If you plan to execute Transformer fine-tuning in Research mode (`Config.ENABLE_TRANSFORMERS = True`), install PyTorch with appropriate CUDA support:
```bash
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
```

---

## 5. MySQL Database Integration & Configuration

The project features a dual database strategy:
1. **Production Mode (MySQL 8+):** Uses SQLAlchemy 2.x and PyMySQL with pooled connections.
2. **Local Fallback (SQLite):** If MySQL environment variables are unset or unreachable, the system automatically logs articles, predictions, and experiment metrics to `outputs/fake_news.db`.

### 1. Configure Credentials
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your MySQL connection details:
```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=fake_news_db
DB_USER=app_user
DB_PASSWORD=your_secure_password
```

### 2. Initialize Database Schema
Run the schema creation script on your MySQL instance:
```bash
mysql -u root -p fake_news_db < sql/schema.sql
```
Alternatively, the notebook and `src/database.py` automatically execute `Base.metadata.create_all(engine)` upon connection!

### Relational Schema Summary:
* `news_articles`: SHA-256 deduplicated article catalog.
* `prediction_history`: Real-time audit log of predictions and probability breakdowns.
* `model_experiments`: Evaluation metrics tracking for all training runs.
* `model_registry`: Versioned model serialization catalog with configuration metadata.
* `users`: Analyst role-based access control with salted SHA-256 password hashing.

---

## 6. How to Run the Primary Deliverable (`Advanced_Fake_News_Detection.ipynb`)

1. Start your Jupyter environment:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
2. Open `Advanced_Fake_News_Detection.ipynb`.
3. Set your execution configuration in **Section 3 (`Config`)**:
   * `EXPERIMENT_MODE = "quick"`: Rapid baseline training on laptop CPU.
   * `EXPERIMENT_MODE = "research"`: Full-corpus grid search and Transformer fine-tuning.
   * `ENABLE_TRANSFORMERS = False` (or `True` if PyTorch + GPU is ready).
4. Run all cells sequentially (**Kernel -> Restart & Run All**).

### Notebook Architecture:
| Section | Topic | Key Deliverable |
| :--- | :--- | :--- |
| **Section 1** | Project Introduction | Ethical framing & scientific truth limitations |
| **Section 2** | Environment Validation | Automated dependency audit table |
| **Section 3** | Configuration & Seeds | Centralized config & hardware accelerator detection |
| **Section 4** | Dataset Ingestion | Schema checks & SHA-256 hash generation |
| **Section 5** | Data Quality Audit | Missingness, empty text detection & 4-panel Seaborn plots |
| **Section 6** | Exploratory Data Analysis | Reuters dateline bias discovery & stylometrics |
| **Section 7** | Duplicate & Leakage Analysis | Zero cross-split leakage enforcement via hash deduplication |
| **Section 8** | Text Preprocessing | Separate ML vs Transformer preprocessing pipelines |
| **Section 9** | Train/Val/Test Splitting | Strict 70/15/15 leak-free stratified partitioning |
| **Section 10** | Feature Engineering | Sublinear TF-IDF + 9 custom stylometric signals |
| **Section 11** | Baseline ML Classifiers | Dummy, Naive Bayes, Logistic Regression, Linear SVC, RF |
| **Section 12** | Advanced TF-IDF Ablation | Title-only vs Body-only vs Combined vs Linguistics |
| **Section 13** | Transformer Classification | DistilBERT fine-tuning pipeline with graceful fallback |
| **Section 14** | Hyperparameter Tuning | 3-Fold RandomizedSearchCV on training partition |
| **Section 15** | Comprehensive Evaluation | Held-out test metrics, ROC & PR Curves, latency audit |
| **Section 16** | Probability Calibration | Reliability curves & Brier score optimization |
| **Section 17** | Explainable AI (XAI) | Linear coefficients & sample-level token attribution |
| **Section 18** | Error & Robustness Analysis | False positive/negative audit & perturbation stability tests |
| **Section 19** | MySQL Database Persistence | Table creation, article hashing & experiment logging |
| **Section 20** | Model Serialization | Joblib pipeline saving & round-trip verification |
| **Section 21** | Production Inference | Real-time `predict_news()` with scientific disclaimer |
| **Section 22** | Research Conclusions | Executive report generation (`outputs/final_report.md`) |

---

## 7. Experimental Benchmark Results (Held-Out Test Set)

Evaluated strictly on the held-out test partition (2,250 articles):

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC | MCC | Inference Latency (per 1k) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVC** | **99.60%** | **99.59%** | **99.51%** | **99.75%** | **0.9963** | **0.9996** | **0.9919** | **0.0009s** |
| **Tuned Logistic Regression** | 99.24% | 99.22% | 99.10% | 99.51% | 0.9930 | 0.9993 | 0.9848 | 0.0010s |
| **Random Forest** | 99.07% | 99.02% | 98.70% | 99.59% | 0.9914 | 0.9994 | 0.9812 | 0.0583s |
| **Logistic Regression (Base)**| 98.84% | 98.81% | 98.62% | 99.26% | 0.9894 | 0.9985 | 0.9767 | 0.0009s |
| **Multinomial Naive Bayes** | 96.36% | 96.34% | 96.79% | 96.48% | 0.9663 | 0.9927 | 0.9266 | 0.0020s |
| **Dummy (Stratified)** | 47.60% | 47.28% | 51.70% | 51.07% | 0.5138 | 0.4728 | -0.0543 | 0.0003s |

### Key Scientific Insights:
1. **The Reuters Dateline Shortcut:** In this dataset, Real articles almost universally originate from Reuters and begin with dateline signatures like `WASHINGTON (Reuters) -`. Linear coefficients demonstrate that words like `"reuters"`, `"spokesman"`, and `"said"` have the highest positive weights. Near-perfect scores reflect this structural artifact.
2. **Calibration Impact:** Raw Logistic Regression exhibited uncalibrated probability clustering. Post-hoc isotonic calibration improved the Brier score from **0.0205** to **0.0118** (42.4% reduction in probabilistic error).
3. **Perturbation Robustness:** The model demonstrated 100% invariance to whitespace injection and case alterations on sample test sets.

---

## 8. Python API & Programmatic Usage

You can also use the modular Python package directly in your applications:

```python
from src.inference import predict_news, explain_prediction

headline = "Global Economic Forum Concludes Annual Summit"
body = "Representatives finalized multi-lateral trade frameworks during today's closing session."

# 1. Real-Time Prediction
prediction = predict_news(headline, body, model_name="fake_news_logistic_pipeline")
print(f"Predicted Class: {prediction['predicted_label']}")
print(f"Confidence: {prediction['confidence_score']:.4f}")
print(f"Notice: {prediction['disclaimer']}")

# 2. Localized Explainability
explanation = explain_prediction(headline, body, model_name="fake_news_logistic_pipeline")
print("Top Contributing Tokens:", explanation["top_contributing_tokens"][:5])
```

---

## 9. License & Citations

Distributed under the MIT License. If using this research or code, please reference the Clément Bisaillon Fake and Real News Dataset on Kaggle.
