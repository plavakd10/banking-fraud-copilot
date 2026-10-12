
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.feature_engineering import engineer_features


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "transactions.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "country_mismatch",
    "card_present",
    "transactions_last_24h",
    "avg_amount_30d",
    "amount_to_average_ratio",
    "is_night_transaction",
    "is_high_amount",
    "is_high_velocity",
]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    df = engineer_features(pd.read_csv(DATA_FILE))
    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_rows": int(len(X_test)),
        "test_fraud_count": int(y_test.sum()),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save preprocessing and classifier together.
    joblib.dump(model, MODEL_DIR / "fraud_baseline.joblib")

    metadata = {
        "model_type": "LogisticRegression",
        "features": FEATURES,
        "threshold": 0.50,
        "data_type": "synthetic",
        "metrics": metrics,
    }

    with open(MODEL_DIR / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    with open(REPORT_DIR / "baseline_metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metadata, indent=2))
    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()
