# 🏦 Enterprise Real-Time Fraud Detection & Financial Crime Monitoring

A state-of-the-art, real-time distributed fraud detection system engineered using a **Hybrid AI Ensemble architecture** (Supervised, Unsupervised, Deep Sequential, and Graph Learning) and organized around a **Medallion Data Lakehouse Architecture** (Bronze ➡️ Silver ➡️ Gold).

---

## 🏛️ Medallion Data Lakehouse Architecture

The system processes and refines transactional data through logical steps:

```
  [ Live Transaction Stream ]
               │
               ▼
   [ Bronze: Raw Storage ]  ──► partitioned JSONL appending of raw API events
               │
               ▼
  [ Silver: Enriched Store ] ──► schema-validated, feature-enriched CSVs (with ML scores)
               │
               ▼
    [ Gold: KPI Aggregates ] ──► high-velocity enterprise aggregates & business KPIs
```

- **Bronze Layer (`bronze/`)**: Appends raw, untouched transaction JSON objects into date-partitioned JSONL storage (`bronze/data/year=YYYY/month=MM/day=DD/raw_transactions.jsonl`). Input validation rejects non-dict and empty payloads. All timestamps are UTC.
- **Silver Layer (`silver/`)**: Validates transaction schemas, enforces data types on boolean/float fields, joins real-time features (Redis velocity, Neo4j degree centrality), appends ML risk prediction flags, and saves to enriched CSVs. Schema drift detection warns when unknown keys appear in incoming data.
- **Gold Layer (`gold/`)**: Executes corporate KPI rollups, computing fraud metrics, category/type breakdowns, and compiling high-risk entity checklists (`gold/data/business_kpis.json`, `gold/data/fraud_by_type.csv`, `gold/data/high_risk_entities.csv`). CSV dtype coercion ensures correct boolean and numeric handling.

---

## 🧠 Hybrid AI Machine Learning Pipeline

Transactions are evaluated in real-time by a multi-model ensembling module located in the `ml/` layer:

1. **Supervised Learning (XGBoost - `xgb_model.pkl`)**: Trained on historical data to recognize established fraudulent behaviors and high-value theft patterns. *Ensemble weight: 40%.*
2. **Unsupervised Learning (Isolation Forest - `iso_model.pkl`)**: Catches zero-day exploits, anomalies, and outliers that deviate from the normal transaction baseline. *Ensemble weight: 25%.*
3. **Deep Sequence Learning (PyTorch LSTM - `lstm_model.pth`)**: Evaluates a rolling sequence of user transactions to identify temporal patterns like **"Smurfing"** (evading $10,000 reporting thresholds through multiple micro-transfers). Uses mini-batch DataLoader training with 80/20 validation split. *Ensemble weight: 35%.*
4. **Graph Network Analytics (Neo4j / NetworkX)**: Computes degree centrality dynamically to identify "Money-Laundering Hubs" (nodes with abnormally high transaction links).

---

## 📁 Unified Directory Structure

```text
Fraud Detection/
├── bronze/                     # Bronze Raw Data Ingestion (Append-only Partitioned JSONL)
│   └── bronze_archiver.py
├── silver/                     # Silver Enriched Storage (Validated Schema & ML Scores)
│   └── silver_transformer.py
├── gold/                       # Gold Aggregates (Enterprise KPIs & Fraud Breakdown)
│   ├── gold_aggregator.py
│   └── data/                   # Compiled corporate KPI JSON and CSV outputs
├── ingestion/                  # Data Generation and Raw Event Stream
│   ├── data_generator.py       # Pre-processes PaySim source CSV into clean training data
│   └── producer_service.py     # High-throughput mock transaction streamer (Kafka Producer)
├── streaming/                  # Stream Processing Core
│   ├── consumer_service.py     # Live Kafka Consumer executing real-time ML inference
│   ├── feature_store.py        # Sub-millisecond Redis cache managing client transaction velocity
│   └── graph_service.py        # Neo4j interface for real-time degree centrality monitoring
├── ml/                         # Machine Learning Models & Registry
│   ├── train_model.py          # Script to train traditional models (XGBoost / Isolation Forest)
│   ├── train_lstm.py           # Script to train deep sequence PyTorch LSTM models
│   ├── evaluate_model.py       # Offline model evaluation (precision, recall, F1, ROC AUC)
│   ├── inference.py            # Ensembled risk engine compiling multi-model evaluations
│   ├── models.py               # PyTorch model architecture definitions (FraudLSTM)
│   └── models_registry/        # Serialized weights, encoders, and historical transaction datasets
├── dashboards/                 # Frontend User Interface
│   ├── dashboard.py            # Premium Streamlit UI (Live Ledger, Graphs, Concept Drift)
│   └── drift_report.py         # Evidently AI data drift report generator
├── tests/                      # Automated Test Suite
│   ├── test_inference.py       # ML inference engine tests (5 tests)
│   ├── test_medallion.py       # Medallion pipeline tests (6 tests)
│   ├── test_drift_report.py    # Evidently drift report tests (2 tests)
│   ├── test_evaluation.py      # Model evaluation tests (1 test)
│   └── test_producer_data.py   # Producer data pipeline tests (1 test)
├── cicd/                       # CI/CD Scripts
│   └── run_tests.bat           # Test runner script
├── orchestration/              # [Future] Workflow orchestration (Airflow / Prefect)
├── terraform/                  # [Future] Infrastructure as Code (AWS / GCP / Azure)
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # Technical architecture reference
│   └── GETTING_STARTED.md      # Step-by-step setup guide
├── docker-compose.yml          # Container configuration (Kafka KRaft, Redis, Neo4j)
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── .gitignore                  # Git ignore rules
└── paysim_data.csv             # Raw 493MB source dataset (ignored by git)
```

---

## 🚀 Quick Start Guide

For a migration-friendly setup path, start with [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md). For a compact system map, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Follow these steps to run the pipeline end-to-end:

### 1. Provision Infrastructure
Deploy containerized Kafka (KRaft), Redis, and Neo4j databases:
```bash
docker compose up -d
```
Ensure all services are running and reporting **healthy**:
```bash
docker ps
```

### 2. Configure Virtual Environment & Install Requirements
Create a local Python virtual environment and install the unified dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate Historical Baseline Training Data
Extract a processed training subset from the local raw PaySim CSV:
```bash
python ingestion/data_generator.py
```

### 4. Train the ML Models
Train the supervised/unsupervised traditional models and the PyTorch sequence model:
```bash
python ml/train_model.py
python ml/train_lstm.py
```
Refresh model evaluation metrics without retraining:
```bash
python ml/evaluate_model.py
```

### 5. Run the Real-Time Streaming Pipeline
Activate the virtual environment across **3 separate terminals**:

*   **Terminal 1 (Core Consumer & Inference Engine)**:
    ```bash
    python streaming/consumer_service.py
    ```
*   **Terminal 2 (High-throughput Ingestion Stream)**:
    ```bash
    python ingestion/producer_service.py
    ```
*   **Terminal 3 (Interactive Management Dashboard)**:
    ```bash
    streamlit run dashboards/dashboard.py
    ```

### 6. Interact with the Dashboard
Navigate to `http://localhost:8501` to view your dashboard. Open the **🔴 Live Stream** tab and click **▶️ Start Stream** to watch transactions stream through the Medallion lakehouse and witness AI ensemble inference live!

Producer modes:

```bash
# Synthetic random stream
set PRODUCER_MODE=random

# Replay PaySim-derived historical transactions with ground-truth labels
set PRODUCER_MODE=paysim_replay

# Replay PaySim data and inject occasional smurfing scenarios
set PRODUCER_MODE=mixed
```

Optional tuning:

```bash
set STREAM_DELAY_SECONDS=0.5
set SMURFING_INJECTION_RATE=0.05
```

---

## 📈 Quality Assurance & Tests

The project includes **15 automated tests** covering Medallion stage transformations, input validation, schema enforcement, ML inference behavior, data drift detection, and model evaluation.

Run the test suite with:
```bash
.venv\Scripts\python -m pytest
```

Or use the CI/CD script:
```bash
cicd\run_tests.bat
```

---

## 🔧 Technical Standards

The codebase follows these production-ready practices:

| Practice | Details |
|----------|---------|
| **Structured Logging** | All modules use Python `logging` (no `print()` statements) |
| **UTC Timestamps** | All partitions and records use `datetime.now(timezone.utc)` |
| **Input Validation** | Bronze and Silver reject non-dict/empty payloads |
| **Data Type Safety** | Silver enforces bool/float types before CSV write; Gold coerces CSV strings back to native types |
| **Thread Safety** | Bronze and Silver file writes are protected by `threading.Lock()` |
| **Modern APIs** | PyTorch `weights_only=True`, Neo4j `count {}`, `redis.Redis`, Python 3 `super()` |
| **Streamlit 1.28+** | Uses `use_container_width=True` instead of deprecated `width='stretch'` |

---

## 🔮 Future Integration Roadmap

The following directories are reserved for planned enhancements. Contributions welcome!

### 🏗️ Infrastructure as Code (`terraform/`)
- **Cloud deployment**: Terraform modules for AWS (EKS + MSK + ElastiCache), GCP (GKE + Pub/Sub + Memorystore), or Azure (AKS + Event Hubs + Azure Cache)
- **Auto-scaling**: Horizontal pod autoscaling for consumer and producer services
- **Secrets management**: Integration with AWS Secrets Manager / GCP Secret Manager / Azure Key Vault

### ⚙️ Workflow Orchestration (`orchestration/`)
- **Airflow / Prefect DAGs**: Scheduled model retraining pipelines with data validation gates
- **Model registry versioning**: MLflow or Weights & Biases integration for experiment tracking
- **Automated Gold refresh**: Scheduled batch aggregation instead of inline Gold compilation

### 🔄 CI/CD Pipeline Expansion (`cicd/`)
- **GitHub Actions / GitLab CI**: Automated test + lint + build on every push
- **Docker image builds**: Multi-stage Dockerfiles for consumer, producer, and dashboard services
- **Deployment automation**: Blue-green or canary deployment strategies for model updates

### 📊 Data & Storage Upgrades
- **Parquet migration**: Replace CSV-based Silver/Gold with Apache Parquet for columnar compression and faster reads
- **Delta Lake / Apache Iceberg**: ACID transactions and time-travel on the Medallion lakehouse
- **PostgreSQL / TimescaleDB**: Persistent relational store for Gold KPIs with SQL query support

### 🤖 ML & MLOps Enhancements
- **Feature Store**: Migrate from Redis to Feast or Tecton for production-grade feature management
- **Online A/B testing**: Shadow-mode deployment to compare new model versions against production
- **Explainability**: SHAP / LIME integration for per-transaction risk explanation
- **Graph Neural Networks**: Upgrade from degree centrality to GNN-based fraud detection (PyG / DGL)
- **Automated retraining**: Drift-triggered retraining pipeline when Evidently detects concept drift

### 🔐 Security & Compliance
- **Authentication**: API gateway (Kong / Envoy) with OAuth2 / JWT for dashboard and services
- **Audit logging**: Immutable audit trail for all fraud decisions (compliance with PCI-DSS / SOX)
- **Data encryption**: At-rest encryption for Bronze/Silver/Gold stores, TLS for all service communication
- **PII masking**: Automatic pseudonymization of account identifiers in non-production environments

### 📡 Monitoring & Observability
- **Prometheus + Grafana**: Real-time metrics for inference latency, throughput, and error rates
- **OpenTelemetry**: Distributed tracing across the Kafka → Consumer → ML → Archive pipeline
- **Alerting**: PagerDuty / Slack integration for anomalous system behavior (latency spikes, consumer lag)
