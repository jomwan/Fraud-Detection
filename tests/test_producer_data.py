import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from ingestion.data_generator import (
    generate_smurfing_sequence,
    iter_paysim_transactions,
    paysim_row_to_transaction,
)


def test_paysim_row_to_transaction_preserves_ground_truth():
    row = pd.Series({
        "step": 1,
        "type": "TRANSFER",
        "amount": 1234.5,
        "nameOrig": "C100",
        "oldbalanceOrg": 2000.0,
        "newbalanceOrig": 765.5,
        "nameDest": "M200",
        "oldbalanceDest": 100.0,
        "newbalanceDest": 1334.5,
        "isFraud": 1,
        "isFlaggedFraud": 0,
    })

    tx = paysim_row_to_transaction(row)

    assert tx["event_source"] == "paysim_replay"
    assert tx["ground_truth_is_fraud"] is True
    assert tx["ground_truth_is_flagged_fraud"] is False
    assert tx["amount"] == 1234.5


def test_iter_paysim_transactions_reads_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([
        {
            "step": 2, "type": "PAYMENT", "amount": 10.0,
            "nameOrig": "C2", "oldbalanceOrg": 100.0, "newbalanceOrig": 90.0,
            "nameDest": "M2", "oldbalanceDest": 0.0, "newbalanceDest": 10.0,
            "isFraud": 0, "isFlaggedFraud": 0,
        },
        {
            "step": 1, "type": "TRANSFER", "amount": 999.0,
            "nameOrig": "C1", "oldbalanceOrg": 999.0, "newbalanceOrig": 0.0,
            "nameDest": "M1", "oldbalanceDest": 0.0, "newbalanceDest": 999.0,
            "isFraud": 1, "isFlaggedFraud": 0,
        },
    ]).to_csv(csv_path, index=False)

    stream = iter_paysim_transactions(str(csv_path), loop=False, chunksize=10)
    first = next(stream)
    second = next(stream)

    assert first["step"] == 1
    assert second["step"] == 2


def test_generate_smurfing_sequence_has_ground_truth_labels():
    seq = generate_smurfing_sequence()

    assert len(seq) == 5
    assert all(tx["event_source"] == "synthetic_smurfing" for tx in seq)
    assert all(tx["ground_truth_is_fraud"] is True for tx in seq)
