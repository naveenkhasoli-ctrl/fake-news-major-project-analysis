"""
Production Inference and Explainability Pipeline.
Handles real-time single article prediction, batch CSV inference,
database logging, and localized feature attribution.
"""

import os
import glob
import logging
from typing import Dict, Any, Optional, List, Union
import joblib
import pandas as pd
import numpy as np

from src.data_processing import generate_article_hash, clean_and_normalize_text
from src.database import save_prediction, get_database_engine

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global cache for loaded model artifacts
_LOADED_MODELS: Dict[str, Any] = {}

SCIENTIFIC_DISCLAIMER = (
    "DISCLAIMER: This output is a dataset-based statistical classification learned from "
    "the training corpus. It does NOT constitute factual verification, investigative journalism, "
    "or definitive proof of truth or falsehood."
)


def load_model_pipeline(model_name: Optional[str] = None, artifacts_dir: str = "artifacts/models") -> Any:
    """
    Load a trained model pipeline from the local artifacts directory.
    Employs caching to avoid reloading pipelines repeatedly.
    """
    global _LOADED_MODELS
    if model_name and model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError(f"Artifacts directory '{artifacts_dir}' does not exist.")

    if model_name:
        target_path = os.path.join(artifacts_dir, f"{model_name}.joblib")
        if not os.path.exists(target_path):
            # Try searching for matching file
            matches = glob.glob(os.path.join(artifacts_dir, f"*{model_name}*.joblib"))
            if matches:
                target_path = matches[0]
            else:
                raise FileNotFoundError(f"Model artifact '{target_path}' not found in {artifacts_dir}")
    else:
        # Default to best model or first available joblib file
        available = glob.glob(os.path.join(artifacts_dir, "*.joblib"))
        if not available:
            raise FileNotFoundError(f"No trained .joblib models found in '{artifacts_dir}'. Train models first.")
        target_path = sorted(available)[0]

    logger.info(f"Loading model artifact from: {target_path}")
    pipeline = joblib.load(target_path)
    actual_name = os.path.splitext(os.path.basename(target_path))[0]
    _LOADED_MODELS[actual_name] = pipeline
    return pipeline


def predict_news(
    title: str,
    text: str,
    model_name: Optional[str] = None,
    log_to_db: bool = True,
    engine: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Classify a single news article with full validation, schema enforcement,
    probability reporting, and optional database audit logging.
    """
    # 1. Input Validation
    clean_title = str(title or "").strip()
    clean_body = str(text or "").strip()
    if not clean_title and not clean_body:
        raise ValueError("Article title and body cannot both be empty.")

    # 2. Representation formatting
    combined_input = f"Title: {clean_title} Article: {clean_body}"
    article_hash = generate_article_hash(clean_title, clean_body)

    # 3. Model Loading
    pipeline = load_model_pipeline(model_name=model_name)
    resolved_model_name = model_name or "best_model"

    # 4. Inference
    raw_pred = pipeline.predict([combined_input])[0]
    pred_label = "REAL" if int(raw_pred) == 1 else "FAKE"

    fake_prob = None
    real_prob = None
    confidence_score = None
    confidence_available = False

    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba([combined_input])[0]
            fake_prob = float(probs[0])
            real_prob = float(probs[1])
            confidence_score = float(max(fake_prob, real_prob))
            confidence_available = True
        except Exception as e:
            logger.debug(f"predict_proba unavailable on pipeline: {e}")
    elif hasattr(pipeline, "decision_function"):
        try:
            score = float(pipeline.decision_function([combined_input])[0])
            confidence_score = score
        except Exception as e:
            logger.debug(f"decision_function unavailable on pipeline: {e}")

    result = {
        "article_hash": article_hash,
        "predicted_label": pred_label,
        "predicted_class_id": int(raw_pred),
        "confidence_available": confidence_available,
        "confidence_score": confidence_score,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "model_name": resolved_model_name,
        "disclaimer": SCIENTIFIC_DISCLAIMER
    }

    # 5. Database Logging
    if log_to_db:
        try:
            db_record = {
                "article_hash": article_hash,
                "model_name": resolved_model_name,
                "model_version": "1.0.0",
                "predicted_label": pred_label,
                "confidence_score": confidence_score if confidence_available else None,
                "fake_probability": fake_prob,
                "real_probability": real_prob
            }
            save_prediction(db_record, engine=engine)
            result["logged_to_database"] = True
        except Exception as e:
            logger.warning(f"Could not log prediction to database: {e}")
            result["logged_to_database"] = False

    return result


def predict_batch_csv(
    csv_path: str,
    output_path: Optional[str] = None,
    model_name: Optional[str] = None,
    log_to_db: bool = False
) -> pd.DataFrame:
    """
    Batch classification of articles from an external CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found at '{csv_path}'")

    df = pd.read_csv(csv_path)
    title_col = next((c for c in df.columns if c.lower() == "title"), None)
    text_col = next((c for c in df.columns if c.lower() == "text"), None)

    if not (title_col and text_col):
        raise ValueError(f"Input CSV must contain 'title' and 'text' columns. Found: {list(df.columns)}")

    pipeline = load_model_pipeline(model_name=model_name)
    combined_texts = [
        f"Title: {str(t).strip()} Article: {str(b).strip()}"
        for t, b in zip(df[title_col], df[text_col])
    ]

    preds = pipeline.predict(combined_texts)
    df["predicted_label"] = ["REAL" if p == 1 else "FAKE" for p in preds]

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(combined_texts)
        df["fake_probability"] = probs[:, 0]
        df["real_probability"] = probs[:, 1]
        df["confidence_score"] = np.max(probs, axis=1)

    df["scientific_disclaimer"] = SCIENTIFIC_DISCLAIMER

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Batch predictions saved to: {output_path}")

    return df


def explain_prediction(
    title: str,
    text: str,
    model_name: Optional[str] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Explain individual prediction by extracting the most influential tokens/n-grams
    from a linear TF-IDF pipeline.
    """
    pipeline = load_model_pipeline(model_name=model_name)
    pred_res = predict_news(title, text, model_name=model_name, log_to_db=False)

    vectorizer = pipeline.named_steps.get("tfidf") or pipeline.named_steps.get("vectorizer")
    classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("clf")

    explanations = []
    if vectorizer and classifier and hasattr(classifier, "coef_"):
        combined_text = f"Title: {title} Article: {text}"
        X_sample = vectorizer.transform([combined_text])
        feature_names = vectorizer.get_feature_names_out()
        coefs = classifier.coef_[0]

        # Non-zero tokens in sample
        nz_indices = X_sample.nonzero()[1]
        token_scores = []
        for idx in nz_indices:
            tfidf_val = X_sample[0, idx]
            weight = coefs[idx] * tfidf_val
            token_scores.append((feature_names[idx], float(weight)))

        # Sort by magnitude of contribution
        token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        explanations = token_scores[:top_k]

    return {
        "article_hash": pred_res["article_hash"],
        "predicted_label": pred_res["predicted_label"],
        "confidence_score": pred_res.get("confidence_score"),
        "top_contributing_tokens": explanations,
        "token_interpretation": "Positive weight pushes prediction toward Real (1); negative weight pushes toward Fake (0).",
        "limitations": "Linear token attributions reflect statistical word correlations in the training dataset, not real-world factual validity."
    }
