# Architecture

This project is a real-time fraud detection prototype organized as a medallion data pipeline.

## Runtime Flow

```text
producer_service.py
  -> Kafka topic: transactions-raw
  -> consumer_service.py
  -> Bronze raw JSONL archive
  -> Redis velocity and sequence features
  -> Neo4j graph centrality features
  -> ml/inference.py ensemble scoring
  -> Silver enriched CSV archive
  -> Kafka topic: fraud-alerts
  -> Gold KPI aggregation
  -> Streamlit dashboard
```

## Layers

### Ingestion

`ingestion/data_generator.py` prepares PaySim data for training and generates individual realtime transactions.

`ingestion/producer_service.py` publishes generated transactions to Kafka. It also injects occasional smurfing-like sequences so the LSTM path can be demonstrated.

The producer supports three modes through `PRODUCER_MODE`:

- `random`: synthetic random transactions.
- `paysim_replay`: rows replayed from `ml/models_registry/historical_transactions.csv`.
- `mixed`: PaySim replay plus configurable synthetic smurfing injections.

PaySim replay and injected fraud scenarios carry `ground_truth_is_fraud`, allowing online prediction audit fields to be computed downstream.

### Bronze

`bronze/bronze_archiver.py` stores raw events exactly as received. Output is partitioned by date:

```text
bronze/data/year=YYYY/month=MM/day=DD/raw_transactions.jsonl
```

### Streaming

`streaming/consumer_service.py` is the main orchestrator. It consumes raw events, enriches them, calls ML inference, archives the result, and publishes fraud decisions.

When an event includes `ground_truth_is_fraud`, the consumer adds `prediction_correct` and `prediction_outcome` fields such as `TRUE_POSITIVE`, `FALSE_POSITIVE`, `FALSE_NEGATIVE`, and `TRUE_NEGATIVE`.

`streaming/feature_store.py` uses Redis for sender velocity and rolling transaction sequences.

`streaming/graph_service.py` uses Neo4j to track sender and receiver relationships and compute normalized degree centrality.

### Machine Learning

`ml/train_model.py` trains:

- XGBoost supervised fraud classifier
- Isolation Forest anomaly detector
- label encoder for transaction type
- graph centrality metadata
- offline evaluation report for model review

`ml/train_lstm.py` trains the PyTorch sequence model for smurfing-like behavior.

`ml/inference.py` combines model scores with velocity and graph rules into a final risk decision.

`ml/evaluate_model.py` evaluates the current registry models and writes:

```text
ml/models_registry/evaluation_report.json
```

The report includes precision, recall, F1, ROC AUC, average precision, and confusion matrices for the supervised and anomaly models.

### Silver

`silver/silver_transformer.py` writes enriched, schema-stable transaction records:

```text
silver/data/year=YYYY/month=MM/day=DD/enriched_transactions.csv
```

### Gold

`gold/gold_aggregator.py` reads Silver data and writes business-facing outputs:

- `gold/data/business_kpis.json`
- `gold/data/fraud_by_type.csv`
- `gold/data/high_risk_entities.csv`

### Dashboard

`dashboards/dashboard.py` is the Streamlit UI. It consumes `fraud-alerts`, displays live decisions, charts model behavior, builds a transaction network view, and can generate a drift report.

The dashboard also reads `ml/models_registry/evaluation_report.json` to show offline training or registry evaluation metrics before the live stream is running.

`dashboards/drift_report.py` owns Evidently report generation. It selects shared numeric columns, cleans invalid values, limits the reference sample for faster rendering, and returns embeddable HTML for the Streamlit dashboard.

## External Services

`docker-compose.yml` defines:

- Kafka on `localhost:29092`
- Redis on `localhost:6379`
- Neo4j on `bolt://localhost:7687`

The unit tests do not require these services.
