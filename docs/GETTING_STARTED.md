# Getting Started

This guide is for a fresh local checkout after migration. It focuses on getting the Python environment and tests working first, then running the optional streaming stack.

## 1. Create a virtual environment

From the project root:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If package installation fails with a network or permission error, retry from a terminal that is allowed to access PyPI.

## 2. Run the tests

```cmd
.venv\Scripts\python -m pytest
```

Expected result:

```text
8 passed
```

The tests validate the core medallion flow and model inference behavior without requiring Kafka, Redis, Neo4j, or Streamlit to be running.

## 3. Prepare training data

The full local PaySim file is expected at:

```text
paysim_data.csv
```

Generate the historical model input:

```cmd
.venv\Scripts\python ingestion\data_generator.py
```

This writes:

```text
ml\models_registry\historical_transactions.csv
```

## 4. Train or refresh models

The repository already contains model artifacts under `ml/models_registry`. To retrain them:

```cmd
.venv\Scripts\python ml\train_model.py
.venv\Scripts\python ml\train_lstm.py
```

`ml/train_model.py` also writes an offline evaluation report:

```text
ml\models_registry\evaluation_report.json
```

To refresh the report from the current saved registry models without retraining:

```cmd
.venv\Scripts\python ml\evaluate_model.py
```

## 5. Run the optional streaming demo

Start infrastructure:

```cmd
docker compose up -d
```

Use three terminals with the virtual environment activated:

```cmd
.venv\Scripts\python streaming\consumer_service.py
```

```cmd
.venv\Scripts\python ingestion\producer_service.py
```

```cmd
.venv\Scripts\streamlit run dashboards\dashboard.py
```

Then open:

```text
http://localhost:8501
```

The **Model Performance** tab shows the offline evaluation report first, followed by live stream behavior once transactions are flowing.

The **MLOps Drift Monitor** tab needs at least 20 valid live rows. The report compares shared numeric columns between the historical baseline and live stream, then embeds the Evidently HTML report in the dashboard.

## Producer modes

By default, the producer uses synthetic random transactions:

```cmd
set PRODUCER_MODE=random
.venv\Scripts\python ingestion\producer_service.py
```

For more realistic model testing, replay the PaySim-derived registry data:

```cmd
set PRODUCER_MODE=paysim_replay
.venv\Scripts\python ingestion\producer_service.py
```

To replay PaySim data and inject occasional smurfing scenarios:

```cmd
set PRODUCER_MODE=mixed
set SMURFING_INJECTION_RATE=0.05
.venv\Scripts\python ingestion\producer_service.py
```

Useful tuning:

```cmd
set STREAM_DELAY_SECONDS=0.5
```

PaySim replay events include `ground_truth_is_fraud`, so the consumer and dashboard can show true positives, false positives, false negatives, and true negatives while streaming.

## Troubleshooting

- `pytest is not recognized`: run `.venv\Scripts\python -m pytest` or reinstall requirements.
- Kafka connection errors: confirm `docker compose up -d` is running and Kafka is reachable on `localhost:29092`.
- Redis or Neo4j connection errors: the code has fallback behavior, but graph and velocity features will be less realistic.
- Model registry missing: run the training steps or restore the files under `ml/models_registry`.
- Drift report fails: let the live stream collect at least 20 rows and confirm `ml/models_registry/historical_transactions.csv` exists.
