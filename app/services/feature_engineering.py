
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional risk signals from transaction features."""

    result = df.copy()

    # Compare the transaction amount with the customer's
    # average transaction amount over the last 30 days.
    result["amount_to_average_ratio"] = (
        result["amount"] / result["avg_amount_30d"].clip(lower=1.0)
    )

    # Simple indicators for potentially unusual activity.
    result["is_night_transaction"] = (
        (result["hour"] >= 0) & (result["hour"] <= 4)
    ).astype(int)

    result["is_high_amount"] = (
        result["amount"] > 500
    ).astype(int)

    result["is_high_velocity"] = (
        result["transactions_last_24h"] >= 8
    ).astype(int)

    return result