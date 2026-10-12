from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT_DIR / "data" / "transactions.csv"

def generate_transactions(
    n_samples: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    amount = np.round(
        rng.lognormal(mean=3.5, sigma=1.2, size=n_samples), 2
    )
    hour = rng.integers(0, 24, size=n_samples)
    merchant_category = rng.choice(
        ["grocery", "retail", "travel", "electronics", "other"],
        size=n_samples,
        p=[0.25, 0.25, 0.10, 0.15, 0.25],
    )
    country_mismatch = rng.binomial(1, 0.04, n_samples)
    card_present = rng.binomial(1, 0.65, n_samples)
    transactions_last_24h = rng.poisson(3, n_samples)
    avg_amount_30d = np.round(
        rng.lognormal(mean=3.2, sigma=0.8, size=n_samples), 2
    )

    # Synthetic risk signals. These are assumptions for a demo,
    # not real-world banking fraud rules.
    risk_score = (
        -4.0
        + 1.8 * country_mismatch
        + 0.8 * (hour <= 4)
        + 0.8 * (amount > 500)
        + 0.6 * (transactions_last_24h >= 8)
        + 0.5 * (amount > 3 * avg_amount_30d)
        + 0.3 * (card_present == 0)
    )

    fraud_probability = 1 / (1 + np.exp(-risk_score))
    is_fraud = rng.binomial(1, fraud_probability)

    df = pd.DataFrame(
        {
            "transaction_id": [
                f"TXN{i:07d}" for i in range(1, n_samples + 1)
            ],
            "amount": amount,
            "hour": hour,
            "merchant_category": merchant_category,
            "country_mismatch": country_mismatch,
            "card_present": card_present,
            "transactions_last_24h": transactions_last_24h,
            "avg_amount_30d": avg_amount_30d,
            "is_fraud": is_fraud,
        }
    )

    return df


if __name__ == "__main__":
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    transactions = generate_transactions()
    transactions.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(transactions):,} synthetic transactions.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nFraud label distribution:")
    print(transactions["is_fraud"].value_counts())
    print("\nFraud rate:")
    print(f"{transactions['is_fraud'].mean():.2%}")
    print("\nSample records:")
    print(transactions.head())