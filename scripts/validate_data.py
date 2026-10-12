from pathlib import Path
import pandas as pd
from app.services.data_validation import validate_transactions

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "transactions.csv"

def main() -> None:
    transactions = pd.read_csv(DATA_FILE)
    validate_transactions(transactions)
    
    print(f"Rows: {len(transactions)}")
    print(f"Columns: {len(transactions.columns)}")
    print(f"Fraud rate: {transactions['is_fraud'].mean():.2%}")

if __name__ == "__main__":
    main()    
