import os
import json
import logging
from datetime import datetime, timezone
import threading

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Thread lock to guarantee thread-safe writes in multi-threaded consumer configurations
_lock = threading.Lock()

def archive_to_bronze(raw_tx: dict):
    """
    Archives raw, untouched transaction json objects into partitioned JSONL storage.
    Path: bronze/data/year=YYYY/month=MM/day=DD/raw_transactions.jsonl
    """
    with _lock:
        try:
            if not isinstance(raw_tx, dict) or not raw_tx:
                logger.warning("Rejected non-dict or empty payload in Bronze archiver")
                return False

            now = datetime.now(timezone.utc)
            year_str = f"year={now.strftime('%Y')}"
            month_str = f"month={now.strftime('%m')}"
            day_str = f"day={now.strftime('%d')}"
            
            partition_path = os.path.join(DATA_DIR, year_str, month_str, day_str)
            os.makedirs(partition_path, exist_ok=True)
            
            file_path = os.path.join(partition_path, "raw_transactions.jsonl")
            
            # Append in append-only JSONL format
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(raw_tx) + "\n")
                f.flush()
                
            return True
        except (TypeError, ValueError) as e:
            logger.error("Bronze archiver serialization error: %s", e)
            return False
        except OSError as e:
            logger.error("Bronze archiver I/O error: %s", e)
            return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Test execution
    test_tx = {"amount": 1200.50, "type": "TRANSFER", "nameOrig": "C123", "nameDest": "M999"}
    archive_to_bronze(test_tx)
    logger.info("Bronze archiving test completed!")
