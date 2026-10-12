
from pathlib import Path

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

BASE_NUMERIC = [
    "amount",
    "hour",
    "country_mismatch",
    "card_present",
    "transactions_last_24h",
    "avg_amount_30d",
]
ENGINEERED_NUMERIC = BASE_NUMERIC + [
    "amount_to_average_ratio",
    "is_night_transaction",
    "is_high_amount",
    "is_high_velocity",
]
CATEGORICAL = ["merchant_category"]


def build_model(numeric_features):
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def main():
    df = pd.read_csv(DATA_FILE)
    df = engineer_features(df)

    y = df["is_fraud"]

    # Use the same split for both experiments.
    X_train, X_temp, y_train, y_temp = train_test_split(
        df, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=42, stratify=y_temp
    )

    experiments = {
        "Original features": BASE_NUMERIC,
        "Engineered features": ENGINEERED_NUMERIC,
    }

    print("\nVALIDATION COMPARISON")
    print(f"{'Feature set':24} {'PR-AUC':>10} {'ROC-AUC':>10}")
    print("-" * 48)

    for name, numeric_features in experiments.items():
        model = build_model(numeric_features)

        model.fit(X_train[numeric_features + CATEGORICAL], y_train)

        probabilities = model.predict_proba(
            X_val[numeric_features + CATEGORICAL]
        )[:, 1]

        print(
            f"{name:24} "
            f"{average_precision_score(y_val, probabilities):10.3f} "
            f"{roc_auc_score(y_val, probabilities):10.3f}"
        )


if __name__ == "__main__":
    main()
