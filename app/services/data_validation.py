import pandas as pd

REQUIRED_COLUMNS = {
    "transaction_id",
    "amount",
    "hour",
    "merchant_category",
    "country_mismatch",
    "card_present",
    "transactions_last_24h",
    "avg_amount_30d",
    "is_fraud",
}

def validate_transactions(df: pd.DataFrame) -> None:
    """Validate the structure and basic quality of transaction data."""

    # 1. Check required columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 2. Check for missing values in required columns
    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Transaction data contains missing values.")

    # 3. Check transaction IDs
    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found.")

    # 4. Check transaction amounts
    if (df["amount"] <= 0).any():
        raise ValueError("Transaction amounts must be positive.")

    if (df["avg_amount_30d"] <= 0).any():
        raise ValueError("Average transaction amounts must be positive.")

    # 5. Check hour boundaries
    if not df["hour"].between(0, 23).all():
        raise ValueError("Transaction hour must be between 0 and 23.")

    # 6. Check binary fields
    for column in ["country_mismatch", "card_present", "is_fraud"]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1.")

    # 7. Check transaction counts
    if (df["transactions_last_24h"] < 0).any():
        raise ValueError("Transaction counts cannot be negative.")

    print("Transaction data validation passed.")
