import pandas as pd
import numpy as np
import random
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAYSIM_PATH = os.path.abspath(os.path.join(BASE_DIR, "../paysim_data.csv"))
HISTORICAL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../ml/models_registry/historical_transactions.csv"))

def load_and_preprocess_paysim(paysim_path=PAYSIM_PATH, sample_size=50000):
    """Load PaySim dataset in its original format — no column renaming."""
    print(f"Loading PaySim dataset from {paysim_path}...")
    if not os.path.exists(paysim_path):
        print(f"[ERROR] PaySim CSV not found at {paysim_path}. Check path.")
        return None
    df = pd.read_csv(paysim_path)
    print(f"  Raw dataset: {len(df)} rows, {df['isFraud'].sum()} fraud cases")

    if len(df) > sample_size:
        fraud_df = df[df['isFraud'] == 1]
        normal_df = df[df['isFraud'] == 0].sample(
            n=sample_size - len(fraud_df), random_state=42
        )
        df = pd.concat([fraud_df, normal_df]).reset_index(drop=True)

    df = df.sort_values('step').reset_index(drop=True)
    os.makedirs(os.path.dirname(HISTORICAL_PATH), exist_ok=True)
    df.to_csv(HISTORICAL_PATH, index=False)

    print(f"Preprocessed {len(df)} transactions ({df['isFraud'].sum()} fraud)")
    print(f"Columns preserved: {list(df.columns)}")
    print(f"Saved to '{HISTORICAL_PATH}'")
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
        "type": tx_type,
        "amount": round(amount, 2),
        "nameOrig": f"C{random.randint(100000000, 2100000000)}",
        "oldbalanceOrg": old_bal_orig,
        "newbalanceOrig": new_bal_orig,
        "nameDest": f"M{random.randint(100000000, 2100000000)}",
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest
    }

if __name__ == "__main__":
    load_and_preprocess_paysim()
