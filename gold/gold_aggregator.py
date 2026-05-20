import os
import glob
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SILVER_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../silver/data'))
GOLD_DATA_DIR = os.path.join(BASE_DIR, 'data')

def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix CSV string-to-native type conversion for critical columns."""
    # is_fraud: CSV reads as string "True"/"False" — convert to proper boolean
    if 'is_fraud' in df.columns:
        df['is_fraud'] = df['is_fraud'].astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    # prediction_correct: same treatment
    if 'prediction_correct' in df.columns:
        df['prediction_correct'] = df['prediction_correct'].astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    # ground_truth_is_fraud: same treatment
    if 'ground_truth_is_fraud' in df.columns:
        df['ground_truth_is_fraud'] = df['ground_truth_is_fraud'].astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    # Numeric fields that may arrive as strings
    for col in ['combined_risk', 'amount', 'processing_ms',
                'supervised_risk', 'unsupervised_risk', 'sequence_risk']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

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
        logger.warning("No Silver files discovered yet. Generating baseline fallback.")
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

        # Coerce CSV string types to proper native types
        df = _coerce_dtypes(df)
        
        # 2. Compile Business KPIs
        total_tx = len(df)
        fraud_df = df[df['is_fraud']]
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
            
        logger.info("Gold layer tables updated successfully using %d source transactions.", total_tx)
        return True
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error("Gold aggregator CSV parsing error: %s", e)
        return False
    except OSError as e:
        logger.error("Gold aggregator I/O error: %s", e)
        return False
    except Exception as e:
        logger.error("Gold aggregator unexpected error: %s", e)
        return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    compile_gold_metrics()
