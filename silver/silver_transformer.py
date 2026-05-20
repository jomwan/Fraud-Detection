import os
import csv
from datetime import datetime
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Thread lock to guarantee thread-safe writes in multi-threaded consumer configurations
_lock = threading.Lock()

# Define standard columns to enforce consistent schema
SCHEMA_COLUMNS = [
    "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", 
    "nameDest", "oldbalanceDest", "newbalanceDest", "velocity_5m",
    "supervised_risk", "unsupervised_risk", "sequence_risk", "combined_risk",
    "is_fraud", "action", "reason", "processing_ms", "timestamp"
]

def archive_to_silver(enriched_tx: dict):
    """
    Archives flat enriched transaction data with model scores to partitioned CSV tables.
    Path: silver/data/year=YYYY/month=MM/day=DD/enriched_transactions.csv
    """
    with _lock:
        try:
            now = datetime.now()
            year_str = f"year={now.strftime('%Y')}"
            month_str = f"month={now.strftime('%m')}"
            day_str = f"day={now.strftime('%d')}"
            
            partition_path = os.path.join(DATA_DIR, year_str, month_str, day_str)
            os.makedirs(partition_path, exist_ok=True)
            
            file_path = os.path.join(partition_path, "enriched_transactions.csv")
            
            # Map enriched dictionary to flat row
            row_dict = enriched_tx.copy()
            # If timestamp is missing, append it
            if "timestamp" not in row_dict:
                row_dict["timestamp"] = now.isoformat()
                
            # Filter and prepare row based on schema
            row = [row_dict.get(col, "") for col in SCHEMA_COLUMNS]
            
            file_exists = os.path.exists(file_path)
            
            # Write with safety
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Write schema header
                    writer.writerow(SCHEMA_COLUMNS)
                writer.writerow(row)
                
            return True
        except Exception as e:
            print(f"[ERROR] Failed to archive to Silver: {e}")
            return False

if __name__ == '__main__':
    # Test execution
    test_enriched = {
        "type": "TRANSFER", "amount": 1200.50, "nameOrig": "C123", "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 3799.50, "nameDest": "M999", "oldbalanceDest": 0.0, "newbalanceDest": 1200.50,
        "velocity_5m": 2, "supervised_risk": 0.05, "unsupervised_risk": 0.12, "sequence_risk": 0.02,
        "combined_risk": 0.06, "is_fraud": False, "action": "ALLOW", "reason": "Normal", "processing_ms": 1.45
    }
    archive_to_silver(test_enriched)
    print("Silver transformer test completed!")
