# Architecture

This project is a real-time fraud detection prototype organized as a medallion data pipeline. All modules use Python's `logging` library and UTC timestamps for production readiness.

## Runtime Flow

```text
producer_service.py
  -> Kafka topic: transactions-raw
  -> consumer_service.py
  -> Bronze raw JSONL archive (input-validated, UTC-partitioned)
  -> Redis velocity and sequence features
  -> Neo4j graph centrality features
  -> ml/inference.py ensemble scoring
  -> Silver enriched CSV archive (type-enforced, schema-validated)
  -> Kafka topic: fraud-alerts
  -> Gold KPI aggregation (dtype-coerced)
  -> Streamlit dashboard
```

## Layers

### Ingestion

`ingestion/data_generator.py` prepares PaySim data for training and generates individual realtime transactions.

`ingestion/producer_service.py` publishes generated transactions to Kafka. It also injects occasional smurfing-like sequences so the LSTM path can be demonstrated. Successful delivery messages log at `DEBUG` level to reduce noise in high-throughput scenarios.

The producer supports three modes through `PRODUCER_MODE`:

- `random`: synthetic random transactions.
- `paysim_replay`: rows replayed from `ml/models_registry/historical_transactions.csv`.
- `mixed`: PaySim replay plus configurable synthetic smurfing injections.

PaySim replay and injected fraud scenarios carry `ground_truth_is_fraud`, allowing online prediction audit fields to be computed downstream.

### Bronze

`bronze/bronze_archiver.py` stores raw events exactly as received. Input validation rejects non-dict and empty payloads before any I/O. Output is partitioned by date using UTC timestamps:

```text
bronze/data/year=YYYY/month=MM/day=DD/raw_transactions.jsonl
```

File writes are protected by `threading.Lock()` and flushed to disk before returning.

### Streaming

`streaming/consumer_service.py` is the main orchestrator. It consumes raw events, injects a UTC timestamp, enriches them, calls ML inference, archives the result, and publishes fraud decisions.

When an event includes `ground_truth_is_fraud`, the consumer adds `prediction_correct` and `prediction_outcome` fields such as `TRUE_POSITIVE`, `FALSE_POSITIVE`, `FALSE_NEGATIVE`, and `TRUE_NEGATIVE`.

`streaming/feature_store.py` uses `redis.Redis` (modern API) for sender velocity and rolling transaction sequences.

`streaming/graph_service.py` uses Neo4j to track sender and receiver relationships and compute normalized degree centrality. Uses the modern `count { (u)--() }` syntax (Neo4j 5.x+) instead of the deprecated `size()` function.

### Machine Learning

`ml/train_model.py` trains:

- XGBoost supervised fraud classifier
- Isolation Forest anomaly detector
- label encoder for transaction type
- graph centrality metadata (using a defensive DataFrame copy to avoid mutation)
- offline evaluation report for model review

`ml/train_lstm.py` trains the PyTorch sequence model for smurfing-like behavior. Uses `DataLoader` for mini-batch training (batch_size=64), an 80/20 train/validation split, and reproducible shuffling (`np.random.seed(42)`).

`ml/inference.py` combines model scores with velocity and graph rules into a final risk decision. Uses `torch.load(weights_only=True)` for safe model deserialization.

`ml/evaluate_model.py` evaluates the current registry models and writes:

```text
ml/models_registry/evaluation_report.json
```

The report includes precision, recall, F1, ROC AUC, average precision, and confusion matrices for the supervised and anomaly models. Timestamps use UTC.

`ml/models.py` defines the `FraudLSTM` architecture using modern Python 3 `super()` syntax.

### Silver

`silver/silver_transformer.py` writes enriched, schema-stable transaction records:

```text
silver/data/year=YYYY/month=MM/day=DD/enriched_transactions.csv
```

Key features:
- **Input validation**: rejects non-dict and empty payloads
- **Type enforcement**: casts boolean and float fields before CSV serialization to prevent downstream string-casting bugs
- **Schema drift detection**: logs a warning when incoming data contains keys not in `SCHEMA_COLUMNS`
- **Flush safety**: writes are flushed to disk before returning

### Gold

`gold/gold_aggregator.py` reads Silver data and writes business-facing outputs:

- `gold/data/business_kpis.json`
- `gold/data/fraud_by_type.csv`
- `gold/data/high_risk_entities.csv`

The aggregator coerces CSV string types to proper native types after reading (e.g., `is_fraud` from string `"True"` to boolean `True`, numeric columns via `pd.to_numeric()`).

### Dashboard

`dashboards/dashboard.py` is the Streamlit UI. It consumes `fraud-alerts`, displays live decisions, charts model behavior, builds a transaction network view, and can generate a drift report. Uses `use_container_width=True` (Streamlit 1.28+) and safe `.astype(bool)` for `is_fraud` filtering.

The dashboard also reads `ml/models_registry/evaluation_report.json` to show offline training or registry evaluation metrics before the live stream is running.

`dashboards/drift_report.py` owns Evidently report generation. It selects shared numeric columns, cleans invalid values, limits the reference sample for faster rendering, and returns embeddable HTML for the Streamlit dashboard.

### Tests

The automated test suite (`tests/`) contains **15 tests** across 5 files:

| File | Tests | Coverage |
|------|-------|----------|
| `test_inference.py` | 5 | ML ensemble scoring, velocity/centrality escalation, smurfing detection |
| `test_medallion.py` | 6 | Bronze/Silver/Gold pipeline, input validation, CSV dtype coercion |
| `test_drift_report.py` | 2 | Evidently drift report generation, minimum row validation |
| `test_evaluation.py` | 1 | Model evaluation report generation |
| `test_producer_data.py` | 1 | PaySim data iterator pipeline |

Tests do not require Kafka, Redis, or Neo4j — they mock data directories and use sandboxed file I/O.

## External Services

`docker-compose.yml` defines:

- Kafka on `localhost:29092` (KRaft mode — no Zookeeper dependency)
- Redis on `localhost:6379`
- Neo4j on `bolt://localhost:7687`

The unit tests do not require these services.

## Future Integration Points

Reserved directories for planned enhancements:

| Directory | Purpose | Status |
|-----------|---------|--------|
| `terraform/` | Infrastructure as Code (AWS / GCP / Azure) | Placeholder |
| `orchestration/` | Workflow orchestration (Airflow / Prefect DAGs) | Placeholder |
| `cicd/` | CI/CD pipeline scripts | `run_tests.bat` implemented |

See the **Future Integration Roadmap** section in [`README.md`](../README.md) for detailed plans.
