import json
import logging
import os
import random
import sys
import time

logger = logging.getLogger(__name__)

# Adjust Python Path to resolve local imports cleanly from workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from confluent_kafka import Producer
from ingestion.data_generator import (
    HISTORICAL_PATH,
    generate_realtime_transaction,
    generate_smurfing_sequence,
    iter_paysim_transactions,
)

# Kafka Configuration
conf = {'bootstrap.servers': os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")}
producer = Producer(conf)
PRODUCER_MODE = os.environ.get("PRODUCER_MODE", "paysim_replay").lower()
STREAM_DELAY_SECONDS = float(os.environ.get("STREAM_DELAY_SECONDS", "0.5"))
SMURFING_INJECTION_RATE = float(os.environ.get("SMURFING_INJECTION_RATE", "0.05"))

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result. """
    if err is not None:
        logger.error('Message delivery failed: %s', err)
    else:
        logger.debug('Message delivered to %s [%s]', msg.topic(), msg.partition())

def publish_transaction(tx, delay_seconds=STREAM_DELAY_SECONDS):
    producer.produce(
        'transactions-raw',
        key=tx['nameOrig'],
        value=json.dumps(tx),
        callback=delivery_report
    )
    producer.poll(0)
    if delay_seconds > 0:
        time.sleep(delay_seconds)

def start_streaming():
    logger.info("Starting Kafka Producer in '%s' mode... Press Ctrl+C to stop.", PRODUCER_MODE)
    paysim_stream = None
    if PRODUCER_MODE in {"paysim_replay", "mixed"}:
        paysim_stream = iter_paysim_transactions(HISTORICAL_PATH, loop=True)
        logger.info("Replaying PaySim transactions from %s", HISTORICAL_PATH)

    try:
        while True:
            if PRODUCER_MODE == "paysim_replay":
                tx = next(paysim_stream)
                publish_transaction(tx)
            elif PRODUCER_MODE == "mixed":
                if random.random() < SMURFING_INJECTION_RATE:
                    logger.info("Injecting synthetic smurfing sequence into PaySim replay...")
                    for smurf_tx in generate_smurfing_sequence():
                        publish_transaction(smurf_tx, delay_seconds=0.2)
                else:
                    tx = next(paysim_stream)
                    publish_transaction(tx)
            else:
                tx = generate_realtime_transaction()
                publish_transaction(tx)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_streaming()
