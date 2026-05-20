import json
import logging
import os
import sys
from datetime import datetime, timezone
from confluent_kafka import Consumer, Producer

logger = logging.getLogger(__name__)

# Adjust Python Path to resolve local imports cleanly from workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streaming.feature_store import get_velocity_feature, get_user_sequence
from streaming.graph_service import update_and_get_graph_features
from ml.inference import predict_fraud
from bronze.bronze_archiver import archive_to_bronze
from silver.silver_transformer import archive_to_silver
from gold.gold_aggregator import compile_gold_metrics

# Kafka Configurations
bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
consumer_conf = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': 'fraud_detection_consumer_group',
    'auto.offset.reset': 'earliest'
}
producer_conf = {
    'bootstrap.servers': bootstrap_servers
}

def start_consumer():
    try:
        consumer = Consumer(consumer_conf)
        producer = Producer(producer_conf)
        consumer.subscribe(['transactions-raw'])
        logger.info("Subscribed to transactions-raw. Consuming events...")
    except Exception as e:
        logger.error("Failed to start Kafka broker clients: %s", e)
        return

    processed_count = 0

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue

            try:
                # 1. Parse Raw Transaction Data
                tx = json.loads(msg.value().decode('utf-8'))
                sender_id = tx['nameOrig']
                receiver_id = tx['nameDest']
                amount = float(tx['amount'])
                tx_type = tx['type']

                # Inject UTC timestamp for downstream archival layers
                tx["timestamp"] = datetime.now(timezone.utc).isoformat()

                # 🥉 Archival Step A: Save raw untouched event to Bronze immediately
                archive_to_bronze(tx)

                # 2. Redis Feature Extraction
                velocity = get_velocity_feature(sender_id)
                tx['velocity_5m'] = velocity
                
                seq_amounts = get_user_sequence(sender_id, amount)

                # 3. Neo4j Graph Centrality Calculations
                sender_deg, receiver_deg = update_and_get_graph_features(sender_id, receiver_id)

                # 4. Multi-Model Ensemble Risk Estimation
                pred_results = predict_fraud(
                    tx_type=tx_type,
                    amount=amount,
                    oldbalanceOrg=tx['oldbalanceOrg'],
                    newbalanceOrig=tx['newbalanceOrig'],
                    oldbalanceDest=tx['oldbalanceDest'],
                    newbalanceDest=tx['newbalanceDest'],
                    seq_amounts=seq_amounts,
                    velocity=velocity,
                    sender_deg=sender_deg,
                    receiver_deg=receiver_deg
                )

                # 5. Enrich Transaction with Predictions
                tx.update(pred_results)
                if "ground_truth_is_fraud" in tx:
                    ground_truth = bool(tx["ground_truth_is_fraud"])
                    tx["prediction_correct"] = bool(tx["is_fraud"] == ground_truth)
                    tx["prediction_outcome"] = (
                        "TRUE_POSITIVE" if tx["is_fraud"] and ground_truth else
                        "FALSE_POSITIVE" if tx["is_fraud"] and not ground_truth else
                        "FALSE_NEGATIVE" if (not tx["is_fraud"]) and ground_truth else
                        "TRUE_NEGATIVE"
                    )

                # 🥈 Archival Step B: Save flattened schema-validated record to Silver
                archive_to_silver(tx)

                # 6. Publish block/allow decision to fraud-alerts
                producer.produce(
                    'fraud-alerts',
                    key=sender_id,
                    value=json.dumps(tx)
                )
                producer.poll(0)

                # 🥇 Operational Step C: Compile business operational KPIs dynamically in Gold
                processed_count += 1
                if processed_count % 10 == 0:
                    compile_gold_metrics()

                # CLI Visual Log
                status = "🔴 BLOCKED" if tx['is_fraud'] else "🟢 ALLOWED"
                logger.info(
                    "[%s] TX %s -> %s | Amt: $%,.2f | Risk: %.3f | Latency: %sms | Reason: %s",
                    status, sender_id, receiver_id, amount,
                    tx['combined_risk'], tx['processing_ms'], tx['reason']
                )

            except Exception as ex:
                logger.error("Failed to process transaction event: %s", ex)

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        producer.flush()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_consumer()
