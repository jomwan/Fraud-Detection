import os
import sys
import pytest

# Ensure PYTHONPATH contains workspace root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from ml.inference import predict_fraud, load_all_models

def test_inference_engine_initialization():
    """Asserts that the models register initializes successfully."""
    load_all_models()
    # Check that predict_fraud is callable
    assert callable(predict_fraud)

def test_normal_transaction_low_risk():
    """Asserts that a standard, low-amount transaction gets cleared with low risk."""
    res = predict_fraud(
        tx_type="PAYMENT",
        amount=150.0,
        oldbalanceOrg=1000.0,
        newbalanceOrig=850.0,
        oldbalanceDest=500.0,
        newbalanceDest=650.0,
        seq_amounts=[0.0, 0.0, 0.0, 0.0, 150.0],
        velocity=1,
        sender_deg=0.0,
        receiver_deg=0.0
    )
    
    assert res["combined_risk"] < 0.5
    assert res["is_fraud"] is False
    assert res["action"] == "ALLOW"
    assert res["reason"] == "Normal"

def test_high_velocity_escalation():
    """Asserts that high transaction rate increases risk score."""
    res_normal = predict_fraud(
        tx_type="PAYMENT",
        amount=150.0,
        oldbalanceOrg=1000.0,
        newbalanceOrig=850.0,
        oldbalanceDest=500.0,
        newbalanceDest=650.0,
        seq_amounts=[0.0, 0.0, 0.0, 0.0, 150.0],
        velocity=1,
        sender_deg=0.0,
        receiver_deg=0.0
    )
    
    res_high = predict_fraud(
        tx_type="PAYMENT",
        amount=150.0,
        oldbalanceOrg=1000.0,
        newbalanceOrig=850.0,
        oldbalanceDest=500.0,
        newbalanceDest=650.0,
        seq_amounts=[0.0, 0.0, 0.0, 0.0, 150.0],
        velocity=6, # High velocity
        sender_deg=0.0,
        receiver_deg=0.0
    )
    
    # Assert that combined risk is exactly 0.3 higher (with rounding up to 1.0)
    assert res_high["combined_risk"] == min(1.0, round(res_normal["combined_risk"] + 0.3, 3))

def test_smurfing_sequence_detection():
    """Asserts that PyTorch LSTM flags continuous transactions just under $10,000."""
    res = predict_fraud(
        tx_type="TRANSFER",
        amount=9850.0,
        oldbalanceOrg=100000.0,
        newbalanceOrig=90150.0,
        oldbalanceDest=0.0,
        newbalanceDest=9850.0,
        # Five transactions in the smurfing bracket ($9K to $10K)
        seq_amounts=[9700.0, 9900.0, 9800.0, 9600.0, 9850.0],
        velocity=5,
        sender_deg=0.0,
        receiver_deg=0.0
    )
    
    # LSTM sequence learning triggers alerts
    assert res["sequence_risk"] > 0.7
    assert res["is_fraud"] is True
    assert res["action"] == "BLOCK"
    assert "Smurfing Sequence" in res["reason"]

def test_graph_centrality_escalation():
    """Asserts that money-laundering node centrality escalates the ensembled score."""
    res_normal = predict_fraud(
        tx_type="PAYMENT",
        amount=150.0,
        oldbalanceOrg=1000.0,
        newbalanceOrig=850.0,
        oldbalanceDest=500.0,
        newbalanceDest=650.0,
        seq_amounts=[0.0, 0.0, 0.0, 0.0, 150.0],
        velocity=1,
        sender_deg=0.0,
        receiver_deg=0.0
    )
    
    res_central = predict_fraud(
        tx_type="PAYMENT",
        amount=150.0,
        oldbalanceOrg=1000.0,
        newbalanceOrig=850.0,
        oldbalanceDest=500.0,
        newbalanceDest=650.0,
        seq_amounts=[0.0, 0.0, 0.0, 0.0, 150.0],
        velocity=1,
        sender_deg=0.06, # High centrality hub
        receiver_deg=0.0
    )
    
    # Assert that combined risk is exactly 0.2 higher (with rounding up to 1.0)
    assert res_central["combined_risk"] == min(1.0, round(res_normal["combined_risk"] + 0.2, 3))
