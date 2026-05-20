import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from ml.evaluate_model import generate_evaluation_report


def test_generate_evaluation_report(tmp_path):
    X = np.array([
        [0.1, 1.0],
        [0.2, 1.1],
        [0.3, 1.2],
        [5.0, 9.0],
        [5.2, 9.1],
        [5.4, 9.2],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    xgb_like_model = LogisticRegression(random_state=42).fit(X, y)
    iso_model = IsolationForest(contamination=0.5, random_state=42).fit(X)
    output_path = tmp_path / "evaluation_report.json"

    report = generate_evaluation_report(
        xgb_model=xgb_like_model,
        iso_model=iso_model,
        X_eval=X,
        y_eval=y,
        output_path=str(output_path),
        dataset_name="unit_test",
    )

    assert output_path.exists()
    assert report["dataset"] == "unit_test"
    assert report["row_count"] == 6
    assert report["fraud_count"] == 3
    assert "xgboost" in report["models"]
    assert "isolation_forest" in report["models"]
    assert report["models"]["xgboost"]["precision"] >= 0.0
    assert "true_positive" in report["models"]["xgboost"]["confusion_matrix"]
