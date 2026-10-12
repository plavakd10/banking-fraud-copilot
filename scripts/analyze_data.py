
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "transactions.csv"
OUTPUT_DIR = ROOT_DIR / "reports" / "figures"


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("BANKING TRANSACTION DATA ANALYSIS")
    print("=" * 55)

    # 1. Dataset overview
    print("\n1. Dataset overview")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate transaction IDs: {df['transaction_id'].duplicated().sum()}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # 2. Fraud class distribution
    print("\n2. Fraud class distribution")
    print(df["is_fraud"].value_counts().sort_index())
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")

    # 3. Numerical statistics
    print("\n3. Transaction amount statistics")
    print(df["amount"].describe().round(2))

    print("\n4. Fraud rate by merchant category")
    merchant_rates = (
        df.groupby("merchant_category")["is_fraud"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    merchant_rates["mean"] = (merchant_rates["mean"] * 100).round(2)
    print(merchant_rates.rename(columns={"mean": "fraud_rate_percent"}))

    # 5. Fraud rate by transaction hour
    hourly_rates = df.groupby("hour")["is_fraud"].mean()

    plt.figure(figsize=(10, 5))
    hourly_rates.plot(kind="bar")
    plt.title("Synthetic Fraud Rate by Transaction Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Fraud Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraud_rate_by_hour.png", dpi=150)
    plt.close()

    # 6. Transaction amount distribution
    plt.figure(figsize=(10, 5))
    df.loc[df["is_fraud"] == 0, "amount"].clip(upper=1000).hist(
        bins=50, alpha=0.65, label="Legitimate"
    )
    df.loc[df["is_fraud"] == 1, "amount"].clip(upper=1000).hist(
        bins=50, alpha=0.65, label="Fraud"
    )
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount (clipped at 1000)")
    plt.ylabel("Number of Transactions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transaction_amount_distribution.png", dpi=150)
    plt.close()

    print("\nCharts saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()