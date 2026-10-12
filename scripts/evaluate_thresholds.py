
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "transactions.csv"

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "country_mismatch",
    "card_present",
    "transactions_last_24h",
    "avg_amount_30d",
]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df["is_fraud"]

    # 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=42, stratify=y_temp
    )

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         CATEGORICAL_FEATURES),
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

    # Choose a threshold using validation data only.
    val_prob = model.predict_proba(X_val)[:, 1]

    print("\nVALIDATION RESULTS")
    print("Threshold | Precision | Recall | F1")
    print("-" * 42)

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        pred = (val_prob >= threshold).astype(int)
        precision = precision_score(y_val, pred, zero_division=0)
        recall = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)

        print(
            f"{threshold:9.2f} | {precision:9.3f} | "
            f"{recall:6.3f} | {f1:.3f}"
        )

    print(f"\nValidation PR-AUC: {average_precision_score(y_val, val_prob):.3f}")

    # Example operating threshold; choose based on validation results
    # and an explicit business objective.
    chosen_threshold = 0.30

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= chosen_threshold).astype(int)

    print(f"\nFINAL TEST RESULTS at threshold {chosen_threshold:.2f}")
    print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, test_pred, zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test, test_pred, zero_division=0):.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, test_pred))


if __name__ == "__main__":
    main()