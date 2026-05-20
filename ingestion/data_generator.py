import logging
import os
import random

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAYSIM_PATH = os.path.abspath(os.path.join(BASE_DIR, "../paysim_data.csv"))
HISTORICAL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../ml/models_registry/historical_transactions.csv"))
PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"
]

def load_and_preprocess_paysim(paysim_path=PAYSIM_PATH, sample_size=50000):
    """Load PaySim dataset in its original format — no column renaming."""
    logger.info("Loading PaySim dataset from %s...", paysim_path)
    if not os.path.exists(paysim_path):
        logger.error("PaySim CSV not found at %s. Check path.", paysim_path)
        return None
    df = pd.read_csv(paysim_path)
    logger.info("  Raw dataset: %d rows, %d fraud cases", len(df), df['isFraud'].sum())

    if len(df) > sample_size:
        fraud_df = df[df['isFraud'] == 1]
        normal_df = df[df['isFraud'] == 0].sample(
            n=sample_size - len(fraud_df), random_state=42
        )
        df = pd.concat([fraud_df, normal_df]).reset_index(drop=True)

    df = df.sort_values('step').reset_index(drop=True)
    os.makedirs(os.path.dirname(HISTORICAL_PATH), exist_ok=True)
    df.to_csv(HISTORICAL_PATH, index=False)

    logger.info("Preprocessed %d transactions (%d fraud)", len(df), df['isFraud'].sum())
    logger.info("Columns preserved: %s", list(df.columns))
    logger.info("Saved to '%s'", HISTORICAL_PATH)
    return df

def generate_realtime_transaction():
    """Generate a single transaction in native PaySim format."""
    tx_type = random.choices(
        ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'],
        weights=[0.40, 0.10, 0.25, 0.20, 0.05]
    )[0]

    if tx_type in ['TRANSFER', 'CASH_OUT']:
        amount = round(random.lognormvariate(8, 2), 2)
    else:
        amount = round(random.lognormvariate(6, 1.5), 2)
    amount = min(amount, 500000)

    old_bal_orig = round(random.uniform(0, 200000), 2)
    new_bal_orig = max(0, round(old_bal_orig - amount, 2))
    old_bal_dest = round(random.uniform(0, 200000), 2)
    new_bal_dest = round(old_bal_dest + amount, 2)

    return {
        "event_source": "synthetic_random",
        "type": tx_type,
        "amount": round(amount, 2),
        "nameOrig": f"C{random.randint(100000000, 2100000000)}",
        "oldbalanceOrg": old_bal_orig,
        "newbalanceOrig": new_bal_orig,
        "nameDest": f"M{random.randint(100000000, 2100000000)}",
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest
    }

def paysim_row_to_transaction(row):
    """Convert a PaySim dataframe row into a streaming transaction payload."""
    tx = {}
    for col in PAYSIM_COLUMNS:
        if col not in row:
            continue
        value = row[col]
        if pd.isna(value):
            value = None
        elif hasattr(value, "item"):
            value = value.item()
        tx[col] = value

    tx["event_source"] = "paysim_replay"
    tx["amount"] = float(tx["amount"])
    tx["oldbalanceOrg"] = float(tx["oldbalanceOrg"])
    tx["newbalanceOrig"] = float(tx["newbalanceOrig"])
    tx["oldbalanceDest"] = float(tx["oldbalanceDest"])
    tx["newbalanceDest"] = float(tx["newbalanceDest"])
    tx["step"] = int(tx.get("step", 0))
    tx["ground_truth_is_fraud"] = bool(int(tx.get("isFraud", 0)))
    tx["ground_truth_is_flagged_fraud"] = bool(int(tx.get("isFlaggedFraud", 0)))
    return tx

def iter_paysim_transactions(source_path=HISTORICAL_PATH, loop=True, chunksize=1000):
    """Yield PaySim transactions in step order for realistic Kafka replay."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"PaySim replay source not found at {source_path}. Run ingestion/data_generator.py first."
        )

    while True:
        for chunk in pd.read_csv(source_path, chunksize=chunksize):
            if "step" in chunk.columns:
                chunk = chunk.sort_values("step")
            for _, row in chunk.iterrows():
                yield paysim_row_to_transaction(row)
        if not loop:
            break

def generate_smurfing_sequence():
    """Create a short synthetic smurfing burst for demo injection."""
    smurf_user = f"C_SMURF_{random.randint(100,999)}"
    old_bal = round(random.uniform(50000, 200000), 2)
    transactions = []

    for _ in range(5):
        amt = round(random.uniform(9500, 9999), 2)
        smurf_tx = {
            "event_source": "synthetic_smurfing",
            "type": "TRANSFER",
            "amount": amt,
            "nameOrig": smurf_user,
            "oldbalanceOrg": old_bal,
            "newbalanceOrig": max(0, round(old_bal - amt, 2)),
            "nameDest": random.choice([
                f"M{random.randint(100000000, 2100000000)}",
                "M_SHELL_001", "M_SHELL_002"
            ]),
            "oldbalanceDest": 0.0,
            "newbalanceDest": amt,
            "ground_truth_is_fraud": True,
            "ground_truth_is_flagged_fraud": False,
        }
        old_bal = smurf_tx["newbalanceOrig"]
        transactions.append(smurf_tx)

    return transactions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_and_preprocess_paysim()
