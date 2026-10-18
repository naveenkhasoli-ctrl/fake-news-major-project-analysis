"""
Data Processing, Cleaning, Audit, and Feature Engineering Pipeline
for Fake News Detection and Misinformation Analysis.
"""

import os
import re
import hashlib
import string
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_article_hash(title: str, text: str) -> str:
    """
    Generate deterministic SHA-256 hash from normalized article title and text.
    """
    normalized = f"{str(title).strip().lower()}|||{str(text).strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_raw_dataset(
    data_dir: str = "data/raw",
    fake_filename: str = "Fake.csv",
    true_filename: str = "True.csv"
) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv from configurable data directory, validate schemas,
    and return combined dataframe with dataset_label (0=Fake, 1=Real).
    """
    # Check data_dir or root directory fallback
    fake_path = os.path.join(data_dir, fake_filename)
    true_path = os.path.join(data_dir, true_filename)

    if not os.path.exists(fake_path) and os.path.exists(fake_filename):
        fake_path = fake_filename
    if not os.path.exists(true_path) and os.path.exists(true_filename):
        true_path = true_filename

    missing = []
    if not os.path.exists(fake_path):
        missing.append(fake_path)
    if not os.path.exists(true_path):
        missing.append(true_path)

    if missing:
        error_msg = (
            f"\n[CRITICAL ERROR] Required dataset files missing: {missing}\n"
            "Please follow one of the workflows below:\n"
            "1. Manual download: Download from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            "   Place 'Fake.csv' and 'True.csv' into 'data/raw/'.\n"
            "2. Kaggle CLI download: kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw/ --unzip\n"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading Fake news dataset from: {fake_path}")
    df_fake = pd.read_csv(fake_path)
    logger.info(f"Loading Real news dataset from: {true_path}")
    df_true = pd.read_csv(true_path)

    required_cols = {"title", "text", "subject", "date"}
    for name, df in [("Fake.csv", df_fake), ("True.csv", df_true)]:
        cols = set(df.columns.str.strip().str.lower())
        if not required_cols.issubset(cols):
            raise ValueError(f"Schema mismatch in {name}: expected columns {required_cols}, found {cols}")

    # Standardize column names
    df_fake.columns = [c.strip().lower() for c in df_fake.columns]
    df_true.columns = [c.strip().lower() for c in df_true.columns]

    # Assign binary labels: 0 = Fake dataset label, 1 = Real dataset label
    df_fake["dataset_label"] = 0
    df_true["dataset_label"] = 1

    df_fake["source_file"] = "Fake.csv"
    df_true["source_file"] = "True.csv"

    combined_df = pd.concat([df_fake, df_true], ignore_index=True)
    
    # Fill NA values in text columns with empty strings
    combined_df["title"] = combined_df["title"].fillna("")
    combined_df["text"] = combined_df["text"].fillna("")
    combined_df["subject"] = combined_df["subject"].fillna("unknown").str.strip()
    combined_df["date"] = combined_df["date"].fillna("unknown").str.strip()

    # Generate unique article hash and ID
    combined_df["article_hash"] = [
        generate_article_hash(t, b) for t, b in zip(combined_df["title"], combined_df["text"])
    ]
    combined_df["article_id"] = [f"ART_{i+1:06d}" for i in range(len(combined_df))]

    logger.info(
        f"Dataset loaded successfully: Total={len(combined_df)} "
        f"(Fake={len(df_fake)}, Real={len(df_true)})"
    )
    return combined_df


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a comprehensive data quality audit without silent deletions.
    """
    total_records = len(df)
    missing_counts = df.isnull().sum().to_dict()
    
    empty_titles = int((df["title"].str.strip() == "").sum())
    empty_texts = int((df["text"].str.strip() == "").sum())
    both_empty = int(((df["title"].str.strip() == "") & (df["text"].str.strip() == "")).sum())
    
    exact_duplicates = int(df.duplicated(subset=["title", "text"]).sum())
    duplicate_titles = int(df.duplicated(subset=["title"]).sum())
    duplicate_texts = int(df.duplicated(subset=["text"]).sum())
    duplicate_hashes = int(df.duplicated(subset=["article_hash"]).sum())

    # Text length metrics
    char_lens = df["text"].str.len()
    word_lens = df["text"].apply(lambda x: len(x.split()))
    
    extremely_short_text = int((word_lens < 10).sum())
    extremely_long_text = int((word_lens > 5000).sum())

    # Date parsing check
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isnull().sum())

    audit_results = {
        "total_records": total_records,
        "missing_counts": missing_counts,
        "empty_titles": empty_titles,
        "empty_texts": empty_texts,
        "both_empty": both_empty,
        "exact_duplicates": exact_duplicates,
        "duplicate_titles": duplicate_titles,
        "duplicate_texts": duplicate_texts,
        "duplicate_hashes": duplicate_hashes,
        "extremely_short_text_count": extremely_short_text,
        "extremely_long_text_count": extremely_long_text,
        "invalid_date_count": invalid_dates,
        "char_len_stats": char_lens.describe().to_dict(),
        "word_len_stats": word_lens.describe().to_dict(),
        "label_distribution": df["dataset_label"].value_counts().to_dict(),
        "subject_distribution": df["subject"].value_counts().to_dict(),
    }
    return audit_results


def clean_and_normalize_text(text: str, remove_reuters_header: bool = False) -> str:
    """
    Normalize text for traditional ML:
    - Normalizes whitespace and newlines
    - Preserves meaningful punctuation tokens
    - Optionally flags or removes systematic Reuters datelines if requested
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if remove_reuters_header:
        # Many Real news articles begin with "WASHINGTON (Reuters) - " or "(Reuters) -"
        text = re.sub(r"^[A-Z\s,]+(?:\(Reuters\)|Reuters)\s*-\s*", "", text, flags=re.IGNORECASE)
    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer extracting stylometric/linguistic signals:
    - Character count
    - Word count
    - Average word length
    - Sentence count
    - Uppercase character ratio
    - Punctuation ratio
    - Exclamation mark count
    - Question mark count
    - Digit ratio
    """
    def __init__(self):
        self.feature_names_ = [
            "char_count",
            "word_count",
            "avg_word_len",
            "sentence_count",
            "uppercase_ratio",
            "punctuation_ratio",
            "exclamation_count",
            "question_count",
            "digit_ratio",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for doc in X:
            s = str(doc)
            char_count = len(s)
            words = s.split()
            word_count = len(words)
            avg_word_len = char_count / word_count if word_count > 0 else 0.0
            
            # Sentence count heuristic
            sentence_count = max(1, len(re.split(r"[.!?]+", s)) - 1)
            
            upper_count = sum(1 for c in s if c.isupper())
            uppercase_ratio = upper_count / char_count if char_count > 0 else 0.0
            
            punct_count = sum(1 for c in s if c in string.punctuation)
            punctuation_ratio = punct_count / char_count if char_count > 0 else 0.0
            
            exclamation_count = s.count("!")
            question_count = s.count("?")
            
            digit_count = sum(1 for c in s if c.isdigit())
            digit_ratio = digit_count / char_count if char_count > 0 else 0.0

            features.append([
                char_count,
                word_count,
                avg_word_len,
                sentence_count,
                uppercase_ratio,
                punctuation_ratio,
                exclamation_count,
                question_count,
                digit_ratio,
            ])
        return np.array(features, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def split_dataset_leak_free(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    deduplicate_first: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict 3-way train/validation/test split.
    Prevents exact duplicate leakage across splits by deduplicating on article_hash
    prior to stratified splitting.
    """
    if deduplicate_first:
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=["article_hash"]).copy()
        dropped = initial_count - len(df_clean)
        if dropped > 0:
            logger.info(f"Deduplicated {dropped} identical articles to ensure zero cross-split leakage.")
    else:
        df_clean = df.copy()

    # Stratified Train+Val vs Test split
    train_val_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        stratify=df_clean["dataset_label"],
        random_state=random_state
    )

    # Stratified Train vs Validation split
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df["dataset_label"],
        random_state=random_state
    )

    logger.info(
        f"Split complete (Seed={random_state}): "
        f"Train={len(train_df)} ({len(train_df)/len(df_clean):.1%}), "
        f"Val={len(val_df)} ({len(val_df)/len(df_clean):.1%}), "
        f"Test={len(test_df)} ({len(test_df)/len(df_clean):.1%})"
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
