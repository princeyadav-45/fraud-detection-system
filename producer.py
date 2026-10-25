
import json
import time
import uuid
import random
from datetime import datetime, timezone

from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    key_serializer=lambda key: key.encode("utf-8"),
)

merchants = ["Amazon", "Flipkart", "Local Store", "Electronics Shop"]
locations = ["Nagpur", "Mumbai", "Pune", "Delhi"]

print("Transaction producer started. Press Ctrl+C to stop.")

try:
    while True:
        transaction = {
            "transaction_id": str(uuid.uuid4())[:8],
            "user_id": f"USER-{random.randint(1, 20):03d}",
            "amount": random.choice(
                [250, 499, 1200, 2500, 5000, 25000, 50000]
            ),
            "merchant": random.choice(merchants),
            "location": random.choice(locations),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        producer.send(
            "transactions",
            key=transaction["transaction_id"],
            value=transaction,
        ).get(timeout=10)

        print("Published:", json.dumps(transaction))
        time.sleep(2)

except KeyboardInterrupt:
    print("\nProducer stopped.")

finally:
    producer.flush()
    producer.close()
