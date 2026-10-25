from dotenv import load_dotenv

load_dotenv()
import json
import os
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer
from fraud_detector import predict_transaction

# Aiven Cloud Kafka connection
KAFKA_HOST = os.getenv("KAFKA_HOST")
KAFKA_PORT = os.getenv("KAFKA_PORT", "15182")
KAFKA_USERNAME = os.getenv("KAFKA_USERNAME")
KAFKA_PASSWORD = os.getenv("KAFKA_PASSWORD")

required = {
    "KAFKA_HOST": KAFKA_HOST,
    "KAFKA_USERNAME": KAFKA_USERNAME,
    "KAFKA_PASSWORD": KAFKA_PASSWORD,
}
missing = [name for name, value in required.items() if not value]

if missing:
    raise RuntimeError(
        "Missing environment variables: " + ", ".join(missing)
    )

BOOTSTRAP_SERVERS = f"{KAFKA_HOST}:{KAFKA_PORT}"

KAFKA_CONFIG = {
    "bootstrap_servers": BOOTSTRAP_SERVERS,
    "security_protocol": "SASL_SSL",
    "ssl_cafile": str(Path(__file__).resolve().parent / "ca.pem"),
    "sasl_mechanism": "PLAIN",
    "sasl_plain_username": KAFKA_USERNAME,
    "sasl_plain_password": KAFKA_PASSWORD,
}

TRANSACTIONS_TOPIC = "transactions"
ALERTS_TOPIC = "fraud-alerts"

DATA_FILE = (
    Path(__file__).parent / "data" / "processed_transactions.jsonl"
)
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

consumer = KafkaConsumer(
    TRANSACTIONS_TOPIC,
    **KAFKA_CONFIG,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="fraud-detection-ml-v2",
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
)

alert_producer = KafkaProducer(
    **KAFKA_CONFIG,
    key_serializer=lambda key: key.encode("utf-8"),
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

print("FraudShield consumer connected to cloud Kafka.")
print("Waiting for transactions... Press Ctrl+C to stop.")

try:
    for message in consumer:
        transaction = message.value

        try:
            result = predict_transaction(transaction)
            result["status"] = str(result["status"]).upper()

            # Temporary local storage; use a shared database for deployment.
            with DATA_FILE.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result) + "\n")

            print(
                f"ID: {result.get('transaction_id')} | "
                f"Amount: Rs.{result.get('amount')} | "
                f"Risk: {float(result['risk_score']):.2%} | "
                f"Status: {result['status']}"
            )

            if result["status"] == "SUSPICIOUS":
                alert_producer.send(
                    ALERTS_TOPIC,
                    key=str(result["transaction_id"]),
                    value=result,
                ).get(timeout=10)

                print("  FRAUD ALERT PUBLISHED")

        except Exception as error:
            print(
                "Error processing transaction "
                f"{transaction.get('transaction_id', 'UNKNOWN')}: {error}"
            )

except KeyboardInterrupt:
    print("\nConsumer stopped.")

finally:
    consumer.close()
    alert_producer.flush()
    alert_producer.close()
    print("Kafka connections closed.")
