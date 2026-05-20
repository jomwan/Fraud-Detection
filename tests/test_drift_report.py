import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from dashboards.drift_report import build_drift_report_html


def test_build_drift_report_html_uses_shared_numeric_columns():
    reference_df = pd.DataFrame({
        "amount": list(range(100)),
        "oldbalanceOrg": list(range(100, 200)),
        "type": ["PAYMENT"] * 100,
    })
    current_df = pd.DataFrame({
        "amount": list(range(50, 80)),
        "oldbalanceOrg": list(range(150, 180)),
        "reason": ["Normal"] * 30,
    })

    result = build_drift_report_html(
        reference_df,
        current_df,
        min_current_rows=20,
        max_reference_rows=50,
    )

    assert result.columns == ["amount", "oldbalanceOrg"]
    assert result.reference_rows == 50
    assert result.current_rows == 30
    assert "<html" in result.html.lower()


def test_build_drift_report_html_rejects_too_few_current_rows():
    reference_df = pd.DataFrame({"amount": list(range(100))})
    current_df = pd.DataFrame({"amount": [1, 2, 3]})

    try:
        build_drift_report_html(reference_df, current_df, min_current_rows=20)
    except ValueError as exc:
        assert "Need at least 20 valid current rows" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for insufficient rows")
