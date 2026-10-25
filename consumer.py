
import json
from pathlib import Path

from kafka import KafkaConsumer, KafkaProducer
from fraud_detector import predict_transaction

BOOTSTRAP_SERVERS = "localhost:9092"
TRANSACTIONS_TOPIC = "transactions"
ALERTS_TOPIC = "fraud-alerts"

DATA_FILE = Path(__file__).parent / "data" / "processed_transactions.jsonl"
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

consumer = KafkaConsumer(
    TRANSACTIONS_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="fraud-detection-ml-v2",
    value_deserializer=lambda b: json.loads(b.decode("utf-8")),
)

alert_producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

print("FraudShield consumer started.")
print("Waiting for transactions... Press Ctrl+C to stop.")

try:
    for message in consumer:
        transaction = message.value

        try:
            result = predict_transaction(transaction)
            result["status"] = str(result["status"]).upper()

            # Save processed result for dashboard and status polling.
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
                    key=str(result["transaction_id"]).encode("utf-8"),
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
