import logging
import os
import time

import joblib
import pandas as pd
import torch

from ml.models import FraudLSTM

logger = logging.getLogger(__name__)

# Define absolute paths relative to this file to resolve model files reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_registry')

# Global variables for models
xgb_model = None
iso_model = None
le_type = None
lstm_model = None

def load_all_models():
    global xgb_model, iso_model, le_type, lstm_model
    try:
        xgb_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
        iso_path = os.path.join(MODELS_DIR, 'iso_model.pkl')
        le_path = os.path.join(MODELS_DIR, 'le_type.pkl')
        lstm_path = os.path.join(MODELS_DIR, 'lstm_model.pth')
        
        xgb_model = joblib.load(xgb_path)
        iso_model = joblib.load(iso_path)
        le_type = joblib.load(le_path)
        
        lstm_model = FraudLSTM(input_size=1, hidden_size=16)
        lstm_model.load_state_dict(torch.load(lstm_path, weights_only=True))
        lstm_model.eval()
        
        logger.info("Machine learning and deep learning models loaded from registry.")
    except Exception as e:
        logger.warning("Inference engine failed to load models: %s", e)

# Load models on startup
load_all_models()

def predict_fraud(tx_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, 
                  seq_amounts, velocity, sender_deg, receiver_deg):
    """
    Evaluates risk metrics using supervised, unsupervised, and deep sequence learning models.
    """
    start_time = time.time()
    
    if xgb_model is None or iso_model is None or lstm_model is None or le_type is None:
        return {
            "supervised_risk": 0.0,
            "unsupervised_risk": 0.0,
            "sequence_risk": 0.0,
            "combined_risk": 0.0,
            "is_fraud": False,
            "action": "ALLOW",
            "reason": "Model registry missing",
            "processing_ms": 0.0
        }

    # 1. Encode transaction type
    try:
        t_enc = le_type.transform([tx_type])[0]
    except ValueError:
        t_enc = 0

    # 2. Formulate tabular features
    features = pd.DataFrame([{
        'amount': float(amount),
        'type_encoded': t_enc,
        'oldbalanceOrg': float(oldbalanceOrg),
        'newbalanceOrig': float(newbalanceOrig),
        'oldbalanceDest': float(oldbalanceDest),
        'newbalanceDest': float(newbalanceDest),
        'sender_degree': float(sender_deg),
        'receiver_degree': float(receiver_deg)
    }])

    # 3. Supervised Model (XGBoost)
    xgb_prob = float(xgb_model.predict_proba(features)[0][1])

    # 4. Unsupervised Model (Isolation Forest)
    iso_score = float(iso_model.decision_function(features)[0])
    iso_risk = max(0.0, min(1.0, 0.5 - (iso_score / 2.0)))

    # 5. Deep Learning Sequence Model (PyTorch LSTM)
    # Clip and normalize sequence
    normalized_seq = [min(x, 10000.0) / 10000.0 for x in seq_amounts]
    seq_tensor = torch.tensor([[[x] for x in normalized_seq]], dtype=torch.float32)
    with torch.no_grad():
        lstm_risk = lstm_model(seq_tensor).item()

    # 6. Ensemble Synthesis
    combined_risk = (0.4 * xgb_prob) + (0.25 * iso_risk) + (0.35 * lstm_risk)

    # 7. Escalation Check
    max_individual = max(xgb_prob, iso_risk, lstm_risk)
    if max_individual > 0.8:
        combined_risk = max(combined_risk, max_individual * 0.85)

    # 8. Rules and Centrality Overrides
    if velocity >= 5:
        combined_risk = min(1.0, combined_risk + 0.3)
    if sender_deg > 0.05 or receiver_deg > 0.05:
        combined_risk = min(1.0, combined_risk + 0.2)

    is_fraud = bool(combined_risk > 0.5)

    # Reason taxonomy
    if is_fraud:
        if lstm_risk > 0.8:
            reason = "Deep Learning: Smurfing Sequence Detected"
        elif velocity >= 5:
            reason = "High Velocity Attack"
        elif sender_deg > 0.05 or receiver_deg > 0.05:
            reason = "Graph Anomaly (Money Laundering Hub)"
        elif xgb_prob > 0.6 and iso_risk > 0.6:
            reason = "Known Fraud Pattern + High Anomaly"
        elif xgb_prob > 0.6:
            reason = "Known Fraud Pattern (Supervised)"
        else:
            reason = "Unknown Anomaly Detected (Unsupervised)"
    else:
        reason = "Normal"

    processing_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "supervised_risk": round(xgb_prob, 3),
        "unsupervised_risk": round(iso_risk, 3),
        "sequence_risk": round(lstm_risk, 3),
        "combined_risk": round(combined_risk, 3),
        "is_fraud": is_fraud,
        "action": "BLOCK" if is_fraud else "ALLOW",
        "reason": reason,
        "processing_ms": processing_ms
    }
