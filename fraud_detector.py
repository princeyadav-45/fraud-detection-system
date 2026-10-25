
from pathlib import Path

import joblib
import pandas as pd


# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"

THRESHOLD = 0.65


# -------------------- LOAD TRAINED MODEL --------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}\n"
        "Please run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)


# -------------------- FRAUD PREDICTION --------------------

def predict_transaction(transaction):
    """
    Predict the risk of a transaction using the trained ML model.
    Expected input keys: amount, merchant, location.
    """

    required_fields = ["amount", "merchant", "location"]

    for field in required_fields:
        if field not in transaction:
            raise ValueError(f"Missing transaction field: {field}")

    features = pd.DataFrame(
        [
            {
                "amount": float(transaction["amount"]),
                "merchant": str(transaction["merchant"]),
                "location": str(transaction["location"]),
            }
        ]
    )

    probabilities = model.predict_proba(features)[0]

    # Find the probability belonging to class 1 (fraud).
    classes = list(model.classes_)

    if 1 in classes:
        fraud_index = classes.index(1)
    else:
        fraud_index = classes.index(True) if True in classes else None

    if fraud_index is None:
        raise ValueError(
            f"Model does not contain a fraud class. Classes: {classes}"
        )

    risk_score = float(probabilities[fraud_index])
    status = "SUSPICIOUS" if risk_score >= THRESHOLD else "NORMAL"

    return {
        **transaction,
        "risk_score": round(risk_score, 4),
        "status": status,
        "decision": (
            "VERIFY_OR_CANCEL"
            if status == "SUSPICIOUS"
            else "CONTINUE_WITH_CAUTION"
        ),
    }
