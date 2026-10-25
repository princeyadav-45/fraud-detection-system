
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

random.seed(42)

merchants = ["Amazon", "Flipkart", "Local Store", "Electronics Shop"]
locations = ["Nagpur", "Mumbai", "Pune", "Delhi"]

rows = []

# Synthetic demo data: deliberately simulated labels.
for _ in range(5000):
    amount = random.randint(100, 50000)
    merchant = random.choice(merchants)
    location = random.choice(locations)

    # Demo label rule, not a real-world fraud label.
    fraud_probability = 0.04
    if amount > 30000:
        fraud_probability += 0.35
    if location == "Delhi" and amount > 20000:
        fraud_probability += 0.20
    if merchant == "Electronics Shop" and amount > 25000:
        fraud_probability += 0.15

    is_fraud = int(random.random() < min(fraud_probability, 0.90))

    rows.append({
        "amount": amount,
        "merchant": merchant,
        "location": location,
        "is_fraud": is_fraud,
    })

df = pd.DataFrame(rows)

X = df[["amount", "merchant", "location"]]
y = df["is_fraud"]

categorical = ["merchant", "location"]
numeric = ["amount"]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", SimpleImputer(strategy="median"), numeric),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions, zero_division=0))

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")

print("Demo model saved to models/fraud_model.pkl")
print("IMPORTANT: Evaluation uses synthetic labels, not real fraud data.")
