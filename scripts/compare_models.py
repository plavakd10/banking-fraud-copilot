
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ])


def main():
    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df["is_fraud"]

    # Identical stratified splits for both models.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=42, stratify=y_temp
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    validation_results = {}

    print("\nVALIDATION MODEL COMPARISON")
    print(
        f"{'Model':24} {'PR-AUC':>8} "
        f"{'ROC-AUC':>8} {'Precision':>11} {'Recall':>8}"
    )
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        # Keep the threshold fixed for a fair initial comparison.
        predictions = (probabilities >= 0.50).astype(int)

        pr_auc = average_precision_score(y_val, probabilities)
        roc_auc = roc_auc_score(y_val, probabilities)

        validation_results[name] = {
            "model": model,
            "pr_auc": pr_auc,
        }

        print(
            f"{name:24} {pr_auc:8.3f} {roc_auc:8.3f} "
            f"{precision_score(y_val, predictions, zero_division=0):11.3f} "
            f"{recall_score(y_val, predictions, zero_division=0):8.3f}"
        )

    # Choose the model using validation PR-AUC, not test results.
    best_name = max(
        validation_results,
        key=lambda name: validation_results[name]["pr_auc"],
    )
    best_model = validation_results[best_name]["model"]

    print(f"\nBest validation model by PR-AUC: {best_name}")

    # Evaluate the selected model on the untouched test split.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.50).astype(int)

    print("\nFINAL TEST RESULTS at threshold 0.50")
    print(f"Model: {best_name}")
    print(
        f"PR-AUC: {average_precision_score(y_test, test_probabilities):.3f}"
    )
    print(
        f"ROC-AUC: {roc_auc_score(y_test, test_probabilities):.3f}"
    )
    print(
        f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.3f}"
    )
    print(
        f"Recall: {recall_score(y_test, test_predictions, zero_division=0):.3f}"
    )
    print(
        f"F1: {f1_score(y_test, test_predictions, zero_division=0):.3f}"
    )


if __name__ == "__main__":
    main()
