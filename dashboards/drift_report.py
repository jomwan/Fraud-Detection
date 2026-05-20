from dataclasses import dataclass

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


@dataclass
class DriftReportResult:
    html: str
    columns: list[str]
    reference_rows: int
    current_rows: int


def _clean_numeric_frame(df, columns):
    clean_df = df[list(columns)].copy()
    for col in columns:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    return clean_df.dropna(how="any")


def build_drift_report_html(
    reference_df,
    current_df,
    candidate_cols=None,
    min_current_rows=20,
    max_reference_rows=10000,
):
    """Build an Evidently data drift HTML report from shared numeric columns."""
    if candidate_cols is None:
        candidate_cols = [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "velocity_5m",
            "supervised_risk",
            "unsupervised_risk",
            "sequence_risk",
            "combined_risk",
        ]

    shared_cols = [
        col
        for col in candidate_cols
        if col in reference_df.columns and col in current_df.columns
    ]

    if not shared_cols:
        raise ValueError("No shared numeric columns are available for drift reporting.")

    reference_clean = _clean_numeric_frame(reference_df, shared_cols)
    current_clean = _clean_numeric_frame(current_df, shared_cols)

    if len(current_clean) < min_current_rows:
        raise ValueError(
            f"Need at least {min_current_rows} valid current rows; found {len(current_clean)}."
        )
    if reference_clean.empty:
        raise ValueError("Reference dataset has no valid rows for the selected columns.")

    if len(reference_clean) > max_reference_rows:
        reference_clean = reference_clean.sample(max_reference_rows, random_state=42)

    snapshot = Report(metrics=[DataDriftPreset()]).run(
        reference_data=reference_clean,
        current_data=current_clean,
    )

    return DriftReportResult(
        html=snapshot.get_html_str(as_iframe=False),
        columns=shared_cols,
        reference_rows=len(reference_clean),
        current_rows=len(current_clean),
    )
