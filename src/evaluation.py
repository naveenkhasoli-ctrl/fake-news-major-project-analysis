"""
Model Evaluation, Calibration, Explainability, and Robustness Testing Module.
Provides comprehensive metrics, reliability analysis, and diagnostic visualizations.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_model_comprehensive(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    model_name: str = "Model",
    training_time_sec: float = 0.0
) -> Dict[str, Any]:
    """
    Computes a complete suite of classification and calibration metrics.
    Includes inference latency and memory-conscious evaluation.
    """
    start_inf = time.perf_counter()
    y_pred = model.predict(X_test)
    inf_duration = time.perf_counter() - start_inf
    latency_per_1000 = (inf_duration / max(1, len(y_test))) * 1000.0

    # Probability / Decision Score extraction
    y_prob = None
    roc_auc = None
    pr_auc = None
    brier = None

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                y_prob = probs[:, 1]
                roc_auc = float(roc_auc_score(y_test, y_prob))
                pr_auc = float(average_precision_score(y_test, y_prob))
                brier = float(brier_score_loss(y_test, y_prob))
        except Exception as e:
            logger.debug(f"predict_proba available but failed for {model_name}: {e}")
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            roc_auc = float(roc_auc_score(y_test, scores))
            pr_auc = float(average_precision_score(y_test, scores))
        except Exception as e:
            logger.debug(f"decision_function failed for {model_name}: {e}")

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_rep = classification_report(y_test, y_pred, target_names=["Fake (0)", "Real (1)"], output_dict=True)

    results = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "matthews_corrcoef": round(mcc, 4),
        "brier_score": round(brier, 4) if brier is not None else None,
        "training_duration_sec": round(training_time_sec, 3),
        "inference_latency_per_1000_sec": round(latency_per_1000, 4),
        "confusion_matrix": cm,
        "classification_report": clf_rep,
    }
    return results


def plot_confusion_matrix_annotated(
    cm: List[List[int]],
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """Plot an annotated seaborn heatmap for the confusion matrix."""
    plt.figure(figsize=(6, 5))
    labels = ["Fake (0)", "Real (1)"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar=False, annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Dataset Label", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """Plot probability calibration reliability diagram with Brier score."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    plt.plot(prob_pred, prob_true, "s-", color="#1f77b4", lw=2, label=f"{model_name} (Brier: {brier:.4f})")
    plt.title(f"Calibration Reliability Curve - {model_name}", fontsize=12, weight="bold")
    plt.xlabel("Mean Predicted Probability (Confidence)", fontsize=11)
    plt.ylabel("Fraction of Positives (Observed Ratio)", fontsize=11)
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def extract_linear_feature_importance(
    pipeline: Any,
    top_n: int = 20
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract top positive (indicative of Real) and negative (indicative of Fake)
    n-gram tokens from a fitted linear model pipeline.
    """
    classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("clf")
    vectorizer = (
        pipeline.named_steps.get("tfidf") or 
        pipeline.named_steps.get("vectorizer") or 
        pipeline.named_steps.get("text_tfidf")
    )
    
    if not (classifier and vectorizer and hasattr(classifier, "coef_")):
        raise ValueError("Pipeline must contain a vectorizer and a linear classifier with coef_ attribute.")

    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]

    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]

    top_positive = [(str(feature_names[i]), float(coefs[i])) for i in top_pos_idx]
    top_negative = [(str(feature_names[i]), float(coefs[i])) for i in top_neg_idx]

    return {
        "top_real_features": top_positive,
        "top_fake_features": top_negative
    }


def evaluate_robustness_perturbations(
    pipeline: Any,
    sample_texts: List[str]
) -> pd.DataFrame:
    """
    Test model robustness under small textual perturbations:
    - Original
    - Extra whitespace
    - Lowercase all characters
    - Punctuation removal
    """
    records = []
    for idx, text in enumerate(sample_texts):
        orig_pred = pipeline.predict([text])[0]
        
        # Perturbation 1: Extra whitespace
        text_ws = "   ".join(text.split())
        pred_ws = pipeline.predict([text_ws])[0]

        # Perturbation 2: Lowercase
        text_lower = text.lower()
        pred_lower = pipeline.predict([text_lower])[0]

        # Perturbation 3: Punctuation removal
        import string
        text_no_punct = text.translate(str.maketrans("", "", string.punctuation))
        pred_no_punct = pipeline.predict([text_no_punct])[0]

        records.append({
            "sample_index": idx,
            "original_pred": int(orig_pred),
            "pred_extra_whitespace": int(pred_ws),
            "pred_lowercase": int(pred_lower),
            "pred_no_punctuation": int(pred_no_punct),
            "whitespace_stable": bool(orig_pred == pred_ws),
            "lowercase_stable": bool(orig_pred == pred_lower),
            "punct_stable": bool(orig_pred == pred_no_punct)
        })
    return pd.DataFrame(records)
