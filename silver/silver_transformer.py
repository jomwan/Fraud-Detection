import os
import csv
import logging
from datetime import datetime, timezone
import threading

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Thread lock to guarantee thread-safe writes in multi-threaded consumer configurations
_lock = threading.Lock()

# Define standard columns to enforce consistent schema
SCHEMA_COLUMNS = [
    "event_source", "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", 
    "nameDest", "oldbalanceDest", "newbalanceDest", "velocity_5m",
    "supervised_risk", "unsupervised_risk", "sequence_risk", "combined_risk",
    "is_fraud", "ground_truth_is_fraud", "prediction_correct", "prediction_outcome",
    "action", "reason", "processing_ms", "timestamp"
]

# Fields that must be written as proper types to prevent CSV string-casting bugs downstream
_BOOL_FIELDS = {"is_fraud", "prediction_correct", "ground_truth_is_fraud"}
_FLOAT_FIELDS = {
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "supervised_risk", "unsupervised_risk",
    "sequence_risk", "combined_risk", "processing_ms",
}
_SCHEMA_SET = set(SCHEMA_COLUMNS)

def _enforce_types(row_dict: dict) -> dict:
    """Cast critical fields to their correct types before CSV serialization."""
    for field in _BOOL_FIELDS:
        if field in row_dict and row_dict[field] != "":
            row_dict[field] = str(bool(row_dict[field]))
    for field in _FLOAT_FIELDS:
        if field in row_dict and row_dict[field] != "":
            try:
                row_dict[field] = float(row_dict[field])
            except (TypeError, ValueError):
                pass
    return row_dict

def archive_to_silver(enriched_tx: dict):
    """
    Archives flat enriched transaction data with model scores to partitioned CSV tables.
    Path: silver/data/year=YYYY/month=MM/day=DD/enriched_transactions.csv
    """
    with _lock:
        try:
            if not isinstance(enriched_tx, dict) or not enriched_tx:
                logger.warning("Rejected non-dict or empty payload in Silver transformer")
                return False

            # Warn on schema drift (unknown keys not in SCHEMA_COLUMNS)
            extra_keys = set(enriched_tx.keys()) - _SCHEMA_SET
            if extra_keys:
                logger.warning("Silver schema drift detected — unknown keys: %s", extra_keys)

            now = datetime.now(timezone.utc)
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

            # Enforce correct data types before CSV serialization
            row_dict = _enforce_types(row_dict)
                
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
                f.flush()
                
            return True
        except (TypeError, ValueError) as e:
            logger.error("Silver transformer serialization error: %s", e)
            return False
        except OSError as e:
            logger.error("Silver transformer I/O error: %s", e)
            return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Test execution
    test_enriched = {
        "type": "TRANSFER", "amount": 1200.50, "nameOrig": "C123", "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 3799.50, "nameDest": "M999", "oldbalanceDest": 0.0, "newbalanceDest": 1200.50,
        "velocity_5m": 2, "supervised_risk": 0.05, "unsupervised_risk": 0.12, "sequence_risk": 0.02,
        "combined_risk": 0.06, "is_fraud": False, "action": "ALLOW", "reason": "Normal", "processing_ms": 1.45
    }
    archive_to_silver(test_enriched)
    logger.info("Silver transformer test completed!")
