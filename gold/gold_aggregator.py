import os
import glob
import json
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SILVER_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../silver/data'))
GOLD_DATA_DIR = os.path.join(BASE_DIR, 'data')

def compile_gold_metrics():
    """
    Scans the Silver partition store, aggregates business KPIs,
    and publishes structured tables inside the Gold layer.
    """
    os.makedirs(GOLD_DATA_DIR, exist_ok=True)
    
    # 1. Discover all partitioned CSV files in Silver
    pattern = os.path.join(SILVER_DATA_DIR, "**/enriched_transactions.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        print("[WARNING] No Silver files discovered yet. Generating baseline fallback.")
        # Default metrics
        default_kpis = {
            "total_transactions_analyzed": 0,
            "total_fraud_blocked": 0,
            "block_rate_percentage": 0.0,
            "total_volume_processed": 0.0,
            "total_fraud_blocked_volume": 0.0,
            "avg_processing_latency_ms": 0.0
        }
        with open(os.path.join(GOLD_DATA_DIR, 'business_kpis.json'), 'w', encoding='utf-8') as f:
            json.dump(default_kpis, f, indent=4)
            
        # Empty DataFrames for other tables
        pd.DataFrame(columns=["type", "transaction_count", "fraud_count", "fraud_rate_percentage", "total_amount"]).to_csv(
            os.path.join(GOLD_DATA_DIR, 'fraud_by_type.csv'), index=False
        )
        pd.DataFrame(columns=["nameOrig", "nameDest", "amount", "combined_risk", "reason", "timestamp"]).to_csv(
            os.path.join(GOLD_DATA_DIR, 'high_risk_entities.csv'), index=False
        )
        return False
        
    try:
        # Load and concat all csv partitions
        df_list = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(df_list, ignore_index=True)
        
        # 2. Compile Business KPIs
        total_tx = len(df)
        fraud_df = df[df['is_fraud'] == True]
        total_fraud = len(fraud_df)
        block_rate = (total_fraud / total_tx * 100) if total_tx > 0 else 0.0
        total_volume = float(df['amount'].sum())
        total_fraud_vol = float(fraud_df['amount'].sum())
        avg_latency = float(df['processing_ms'].mean()) if 'processing_ms' in df.columns else 0.0
        
        kpis = {
            "total_transactions_analyzed": total_tx,
            "total_fraud_blocked": total_fraud,
            "block_rate_percentage": round(block_rate, 2),
            "total_volume_processed": round(total_volume, 2),
            "total_fraud_blocked_volume": round(total_fraud_vol, 2),
            "avg_processing_latency_ms": round(avg_latency, 2)
        }
        
        with open(os.path.join(GOLD_DATA_DIR, 'business_kpis.json'), 'w', encoding='utf-8') as f:
            json.dump(kpis, f, indent=4)
            
        # 3. Compile Fraud by Transaction Type
        if 'type' in df.columns:
            type_grp = df.groupby('type').agg(
                transaction_count=('amount', 'count'),
                fraud_count=('is_fraud', lambda x: int(x.sum())),
                total_amount=('amount', 'sum')
            ).reset_index()
            
            type_grp['fraud_rate_percentage'] = (type_grp['fraud_count'] / type_grp['transaction_count'] * 100).round(2)
            type_grp.to_csv(os.path.join(GOLD_DATA_DIR, 'fraud_by_type.csv'), index=False)
            
        # 4. Compile High Risk Entities
        if 'combined_risk' in df.columns:
            high_risk = df[df['combined_risk'] > 0.5].sort_values('combined_risk', ascending=False)
            keep_cols = [c for c in ["nameOrig", "nameDest", "amount", "combined_risk", "reason", "timestamp"] if c in df.columns]
            high_risk = high_risk[keep_cols].head(20)
            high_risk.to_csv(os.path.join(GOLD_DATA_DIR, 'high_risk_entities.csv'), index=False)
            
        print(f"[SUCCESS] Gold layer tables updated successfully using {total_tx} source transactions.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to compile Gold metrics: {e}")
        return False

if __name__ == '__main__':
    compile_gold_metrics()
