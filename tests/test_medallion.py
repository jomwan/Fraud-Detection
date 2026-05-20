import os
import sys
import json
import shutil
import pandas as pd
import pytest

# Ensure PYTHONPATH contains workspace root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

from bronze.bronze_archiver import archive_to_bronze
from silver.silver_transformer import archive_to_silver
from gold.gold_aggregator import compile_gold_metrics

# Use dedicated test data directory within tests
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")

@pytest.fixture(autouse=True)
def setup_test_directories(monkeypatch):
    """Dynamically routes all module data directories to the sandbox tests/test_data/ directory."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Patch paths to point to test sandbox instead of production data stores
    monkeypatch.setattr("bronze.bronze_archiver.DATA_DIR", os.path.join(TEST_DATA_DIR, "bronze"))
    monkeypatch.setattr("silver.silver_transformer.DATA_DIR", os.path.join(TEST_DATA_DIR, "silver"))
    monkeypatch.setattr("gold.gold_aggregator.SILVER_DATA_DIR", os.path.join(TEST_DATA_DIR, "silver"))
    monkeypatch.setattr("gold.gold_aggregator.GOLD_DATA_DIR", os.path.join(TEST_DATA_DIR, "gold"))
    
    yield
    
    # Tear down test data directory
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)

def test_bronze_archiver():
    """Asserts that raw data is successfully logged in append-only JSONL files."""
    test_tx = {"amount": 25000.0, "type": "CASH_OUT", "nameOrig": "C1001", "nameDest": "M2002"}
    
    # Act
    success = archive_to_bronze(test_tx)
    assert success is True
    
    # Assert file existence
    bronze_dir = os.path.join(TEST_DATA_DIR, "bronze")
    assert os.path.exists(bronze_dir)
    
    # Discover JSONL file recursively
    found_files = []
    for root, dirs, files in os.walk(bronze_dir):
        for file in files:
            if file == "raw_transactions.jsonl":
                found_files.append(os.path.join(root, file))
                
    assert len(found_files) == 1
    
    # Validate content
    with open(found_files[0], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert len(lines) == 1
        logged_tx = json.loads(lines[0].strip())
        assert logged_tx["amount"] == 25000.0
        assert logged_tx["type"] == "CASH_OUT"

def test_silver_transformer():
    """Asserts that enriched data is correctly parsed, schema validated, and saved in CSVs."""
    test_enriched = {
        "type": "CASH_OUT", "amount": 25000.0, "nameOrig": "C1001", "oldbalanceOrg": 30000.0,
        "newbalanceOrig": 5000.0, "nameDest": "M2002", "oldbalanceDest": 0.0, "newbalanceDest": 25000.0,
        "velocity_5m": 1, "supervised_risk": 0.92, "unsupervised_risk": 0.81, "sequence_risk": 0.15,
        "combined_risk": 0.89, "is_fraud": True, "action": "BLOCK", "reason": "High Score Overrides", "processing_ms": 12.5
    }
    
    # Act
    success = archive_to_silver(test_enriched)
    assert success is True
    
    # Assert file existence
    silver_dir = os.path.join(TEST_DATA_DIR, "silver")
    assert os.path.exists(silver_dir)
    
    # Discover CSV file
    found_files = []
    for root, dirs, files in os.walk(silver_dir):
        for file in files:
            if file == "enriched_transactions.csv":
                found_files.append(os.path.join(root, file))
                
    assert len(found_files) == 1
    
    # Load and validate columns
    df = pd.read_csv(found_files[0])
    assert len(df) == 1
    assert list(df.columns) == [
        "event_source", "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", 
        "nameDest", "oldbalanceDest", "newbalanceDest", "velocity_5m",
        "supervised_risk", "unsupervised_risk", "sequence_risk", "combined_risk",
        "is_fraud", "ground_truth_is_fraud", "prediction_correct", "prediction_outcome",
        "action", "reason", "processing_ms", "timestamp"
    ]
    assert df.loc[0, "type"] == "CASH_OUT"
    assert bool(df.loc[0, "is_fraud"]) is True
    assert df.loc[0, "combined_risk"] == 0.89

def test_gold_aggregation_compilation():
    """Asserts that Silver files can be aggregated into corporate KPI rollups."""
    # Write mock Silver CSVs
    test_enriched_1 = {
        "type": "CASH_OUT", "amount": 25000.0, "nameOrig": "C1001", "oldbalanceOrg": 30000.0,
        "newbalanceOrig": 5000.0, "nameDest": "M2002", "oldbalanceDest": 0.0, "newbalanceDest": 25000.0,
        "velocity_5m": 1, "supervised_risk": 0.92, "unsupervised_risk": 0.81, "sequence_risk": 0.15,
        "combined_risk": 0.89, "is_fraud": True, "action": "BLOCK", "reason": "High Score Overrides", "processing_ms": 10.0
    }
    test_enriched_2 = {
        "type": "PAYMENT", "amount": 500.0, "nameOrig": "C500", "oldbalanceOrg": 2000.0,
        "newbalanceOrig": 1500.0, "nameDest": "M600", "oldbalanceDest": 0.0, "newbalanceDest": 500.0,
        "velocity_5m": 1, "supervised_risk": 0.02, "unsupervised_risk": 0.05, "sequence_risk": 0.01,
        "combined_risk": 0.03, "is_fraud": False, "action": "ALLOW", "reason": "Normal", "processing_ms": 5.0
    }
    
    # Archive both to silver sandbox
    archive_to_silver(test_enriched_1)
    archive_to_silver(test_enriched_2)
    
    # Compile Gold
    success = compile_gold_metrics()
    assert success is True
    
    # Assert gold targets
    gold_dir = os.path.join(TEST_DATA_DIR, "gold")
    assert os.path.exists(os.path.join(gold_dir, "business_kpis.json"))
    assert os.path.exists(os.path.join(gold_dir, "fraud_by_type.csv"))
    assert os.path.exists(os.path.join(gold_dir, "high_risk_entities.csv"))
    
    # Check KPI metrics
    with open(os.path.join(gold_dir, "business_kpis.json"), 'r', encoding='utf-8') as f:
        kpis = json.load(f)
        assert kpis["total_transactions_analyzed"] == 2
        assert kpis["total_fraud_blocked"] == 1
        assert kpis["block_rate_percentage"] == 50.0
        assert kpis["total_volume_processed"] == 25500.0
        assert kpis["total_fraud_blocked_volume"] == 25000.0
        assert kpis["avg_processing_latency_ms"] == 7.5
        
    # Check Type breakdown
    type_df = pd.read_csv(os.path.join(gold_dir, "fraud_by_type.csv"))
    assert len(type_df) == 2
    assert "CASH_OUT" in type_df["type"].values
    assert "PAYMENT" in type_df["type"].values
    
    # Check high-risk entities
    entities_df = pd.read_csv(os.path.join(gold_dir, "high_risk_entities.csv"))
    assert len(entities_df) == 1
    assert entities_df.loc[0, "nameOrig"] == "C1001"
    assert entities_df.loc[0, "combined_risk"] == 0.89
