
from pathlib import Path
import json
import sys
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from kafka import KafkaProducer
from kafka.errors import KafkaError

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed_transactions.jsonl"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fraud_detector import predict_transaction

BOOTSTRAP_SERVERS = "localhost:9092"
TRANSACTIONS_TOPIC = "transactions"

app = FastAPI(
    title="FraudShield API",
    description="Transaction risk assessment and processing API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

kafka_producer = None


def get_kafka_producer():
    global kafka_producer

    if kafka_producer is None:
        kafka_producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
            key_serializer=lambda key: key.encode("utf-8"),
        )

    return kafka_producer


class TransactionRequest(BaseModel):
    amount: float = Field(gt=0, le=100000000)
    merchant: str = Field(min_length=1, max_length=100)
    location: str = Field(min_length=1, max_length=100)
    user_id: str = Field(default="DEMO-USER", min_length=1, max_length=100)


def find_processed_transaction(transaction_id: str):
    if not DATA_FILE.exists():
        return None

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                    if record.get("transaction_id") == transaction_id:
                        return record
                except json.JSONDecodeError:
                    continue
    except OSError:
        return None

    return None


@app.get("/")
def home():
    return {
        "service": "FraudShield API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    model_path = BASE_DIR / "models" / "fraud_model.pkl"

    return {
        "status": "healthy" if model_path.exists() else "model_missing",
        "model_file_exists": model_path.exists(),
        "kafka_configured": True,
    }


# Existing synchronous prediction endpoint.
# Kept for compatibility with the current dashboard.
@app.post("/predict")
def predict(request: TransactionRequest):
    transaction = {
        "transaction_id": f"APP-{uuid.uuid4().hex[:10].upper()}",
        "user_id": request.user_id.strip(),
        "amount": request.amount,
        "merchant": request.merchant.strip(),
        "location": request.location.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not transaction["merchant"] or not transaction["location"]:
        raise HTTPException(
            status_code=422,
            detail="Merchant and location cannot be empty.",
        )

    try:
        result = predict_transaction(transaction)
        score = float(result["risk_score"])

        if not 0 <= score <= 1:
            raise ValueError("Risk score must be between 0 and 1.")

        status = str(result["status"]).upper()

        return {
            **result,
            "risk_score": round(score, 4),
            "status": status,
            "message": (
                "Suspicious activity flagged. Verify the details."
                if status == "SUSPICIOUS"
                else "No high-risk signal detected by the current demo model."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc


# New asynchronous submission endpoint.
@app.post("/transactions", status_code=202)
def submit_transaction(request: TransactionRequest):
    merchant = request.merchant.strip()
    location = request.location.strip()
    user_id = request.user_id.strip()

    if not merchant or not location or not user_id:
        raise HTTPException(
            status_code=422,
            detail="Merchant, location and user ID cannot be empty.",
        )

    transaction = {
        "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
        "user_id": user_id,
        "amount": request.amount,
        "merchant": merchant,
        "location": location,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        producer = get_kafka_producer()
        producer.send(
            TRANSACTIONS_TOPIC,
            key=transaction["transaction_id"],
            value=transaction,
        ).get(timeout=10)

        return {
            **transaction,
            "processing_status": "PROCESSING",
            "message": "Transaction accepted for risk analysis.",
        }

    except (KafkaError, OSError, TimeoutError, Exception) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not submit transaction to Kafka. "
                f"Check that Kafka is running. Error: {exc}"
            ),
        ) from exc


# Poll this endpoint until the consumer saves the prediction.
@app.get("/transactions/{transaction_id}")
def get_transaction_status(transaction_id: str):
    result = find_processed_transaction(transaction_id)

    if result is None:
        return {
            "transaction_id": transaction_id,
            "processing_status": "PROCESSING",
            "message": "Waiting for the Kafka consumer to process this transaction.",
        }

    status = str(result.get("status", "UNKNOWN")).upper()

    return {
        **result,
        "processing_status": "COMPLETED",
        "message": (
            "Suspicious activity flagged. Verify the transaction details."
            if status == "SUSPICIOUS"
            else "No high-risk signal detected by the current demo model."
        ),
    }


@app.on_event("shutdown")
def shutdown_event():
    global kafka_producer

    if kafka_producer is not None:
        kafka_producer.flush()
        kafka_producer.close()
        kafka_producer = None
