import json
import logging
import os
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODELS_DIR = os.path.join(BASE_DIR, "models_registry")
REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")


def _safe_score(metric_fn, y_true, y_pred_or_score):
    try:
        return round(float(metric_fn(y_true, y_pred_or_score)), 4)
    except Exception:
        return None


def _build_binary_metrics(y_true, y_pred, y_score=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    metrics = {
        "accuracy": _safe_score(accuracy_score, y_true, y_pred),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }

    if y_score is not None:
        metrics["roc_auc"] = _safe_score(roc_auc_score, y_true, y_score)
        metrics["average_precision"] = _safe_score(average_precision_score, y_true, y_score)

    return metrics


def generate_evaluation_report(
    xgb_model,
    iso_model,
    X_eval,
    y_eval,
    output_path=REPORT_PATH,
    dataset_name="holdout",
):
    """Generate supervised and anomaly model evaluation metrics as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_eval = pd.Series(y_eval).astype(int)

    xgb_scores = xgb_model.predict_proba(X_eval)[:, 1]
    xgb_preds = (xgb_scores >= 0.5).astype(int)

    iso_scores = iso_model.decision_function(X_eval)
    iso_risk = [max(0.0, min(1.0, 0.5 - (float(score) / 2.0))) for score in iso_scores]
    iso_preds = [1 if risk >= 0.5 else 0 for risk in iso_risk]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "row_count": int(len(y_eval)),
        "fraud_count": int(y_eval.sum()),
        "fraud_rate": round(float(y_eval.mean()), 4) if len(y_eval) else 0.0,
        "thresholds": {
            "xgboost_fraud_probability": 0.5,
            "isolation_forest_risk": 0.5,
        },
        "models": {
            "xgboost": _build_binary_metrics(y_eval, xgb_preds, xgb_scores),
            "isolation_forest": _build_binary_metrics(y_eval, iso_preds, iso_risk),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return report


def evaluate_registry_models():
    """Evaluate saved registry models against the available historical dataset."""
    from ml.train_model import build_graph_features, preprocess_data

    historical_path = os.path.join(MODELS_DIR, "historical_transactions.csv")
    if not os.path.exists(historical_path):
        raise FileNotFoundError(f"Historical data not found at {historical_path}")

    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    iso_model = joblib.load(os.path.join(MODELS_DIR, "iso_model.pkl"))

    df = pd.read_csv(historical_path)
    df, _, _ = build_graph_features(df)
    X, y, _ = preprocess_data(df)

    return generate_evaluation_report(
        xgb_model=xgb_model,
        iso_model=iso_model,
        X_eval=X,
        y_eval=y,
        dataset_name="registry_historical_dataset",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluation = evaluate_registry_models()
    logger.info("Evaluation report written to %s", REPORT_PATH)
    logger.info(json.dumps(evaluation["models"], indent=4))
