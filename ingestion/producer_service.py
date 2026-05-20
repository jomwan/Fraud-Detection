import json
import time
import sys
import os
import random

# Adjust Python Path to resolve local imports cleanly from workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from confluent_kafka import Producer
from ingestion.data_generator import generate_realtime_transaction

# Kafka Configuration
conf = {'bootstrap.servers': os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")}
producer = Producer(conf)

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result. """
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def start_streaming():
    print("Starting Kafka Producer... Press Ctrl+C to stop.")

    try:
        while True:
            # 5% chance to trigger a Smurfing Attack Sequence!
            if random.random() < 0.05:
                print("⚠️ INJECTING DEEP LEARNING SMURFING SEQUENCE...")
                smurf_user = f"C_SMURF_{random.randint(100,999)}"
                old_bal = round(random.uniform(50000, 200000), 2)
                # Blast 5 transactions just under $10,000 quickly
                for _ in range(5):
                    amt = round(random.uniform(9500, 9999), 2)
                    smurf_tx = {
                        "type": "TRANSFER",
                        "amount": amt,
                        "nameOrig": smurf_user,
                        "oldbalanceOrg": old_bal,
                        "newbalanceOrig": max(0, round(old_bal - amt, 2)),
                        "nameDest": random.choice([
                            f"M{random.randint(100000000, 2100000000)}",
                            "M_SHELL_001", "M_SHELL_002"
                        ]),
                        "oldbalanceDest": 0.0,
                        "newbalanceDest": amt
                    }
                    old_bal = smurf_tx["newbalanceOrig"]
                    producer.produce(
                        'transactions-raw',
                        key=smurf_user,
                        value=json.dumps(smurf_tx),
                        callback=delivery_report
                    )
                    producer.poll(0)
                    time.sleep(0.2) # Fast burst
            else:
                # Generate a normal single transaction
                tx = generate_realtime_transaction()

                # Publish to Kafka
                producer.produce(
                    'transactions-raw',
                    key=tx['nameOrig'],
                    value=json.dumps(tx),
                    callback=delivery_report
                )
                producer.poll(0)

            time.sleep(1) # Simulate standard velocity
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()

if __name__ == '__main__':
    start_streaming()
