import os
import json
from datetime import datetime
import threading

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
            now = datetime.now()
            year_str = f"year={now.strftime('%Y')}"
            month_str = f"month={now.strftime('%m')}"
            day_str = f"day={now.strftime('%d')}"
            
            partition_path = os.path.join(DATA_DIR, year_str, month_str, day_str)
            os.makedirs(partition_path, exist_ok=True)
            
            file_path = os.path.join(partition_path, "raw_transactions.jsonl")
            
            # Append in append-only JSONL format
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(raw_tx) + "\n")
                
            return True
        except Exception as e:
            print(f"[ERROR] Failed to archive to Bronze: {e}")
            return False

if __name__ == '__main__':
    # Test execution
    test_tx = {"amount": 1200.50, "type": "TRANSFER", "nameOrig": "C123", "nameDest": "M999"}
    archive_to_bronze(test_tx)
    print("Bronze archiving test completed!")
