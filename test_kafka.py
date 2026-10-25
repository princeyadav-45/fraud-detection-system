
import logging
import socket
from pathlib import Path
from dotenv import dotenv_values
from kafka import KafkaAdminClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s: %(message)s"
)

env = dotenv_values(Path(__file__).resolve().parent / ".env")
host = env.get("KAFKA_HOST")
port = int(env.get("KAFKA_PORT", "15182"))

print("Host configured:", bool(host))
print("Credentials loaded:", all(
    env.get(k) for k in [
        "KAFKA_USERNAME", "KAFKA_PASSWORD"
    ]
))

try:
    with socket.create_connection((host, port), timeout=10):
        print("TCP connection: OK")
except Exception as e:
    print("TCP connection failed:", repr(e))
    raise SystemExit(1)

client = None
try:
    client = KafkaAdminClient(
        bootstrap_servers=f"{host}:{port}",
        security_protocol="SASL_SSL",
        ssl_cafile=str(Path(__file__).resolve().parent / "ca.pem"),
        sasl_mechanism="PLAIN",
        sasl_plain_username=env.get("KAFKA_USERNAME"),
        sasl_plain_password=env.get("KAFKA_PASSWORD"),
        request_timeout_ms=30000,
    )
    print("Kafka connection successful!")
    print("Topics:", sorted(client.list_topics()))

except Exception as e:
    print("Kafka error type:", type(e).__name__)
    print("Kafka error details:", repr(e))

finally:
    if client is not None:
        client.close()
