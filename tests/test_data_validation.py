import pandas as pd
import pytest

from app.services.data_validation import validate_transactions

@pytest.fixture
def valid_transactions():
    return pd.DataFrame(
        {
            "transaction_id": ["TXN0000001","TXN0000002"],
            "amount": [100.0,250.0],
            "hour": [10,22],
            "merchant_category":["grocery","retail"],
            "country_mismatch":[0,1],
            "card_present":[1,0],
            "transactions_last_24h": [2,5],
            "avg_amount_30d":[120.0,200.0],
            "is_fraud":[0,1]            
        }
    )
    
def test_valid_transactions_pass(valid_transactions):
    validate_transactions(valid_transactions)

def test_navigate_amount_fails(valid_transactions):
    valid_transactions.loc[0,"amount"] = -100
    
    with pytest.raises(ValueError, match = "Transaction amounts must be positive."):
        validate_transactions(valid_transactions)
        
def test_duplicate_transaction_ids_fail(valid_transactions):
    valid_transactions.loc[1,"transaction_id"] = "TXN0000001"
    
    with pytest.raises(ValueError, match = "Duplicate transaction IDs found."):
        validate_transactions(valid_transactions)

def test_missing_column_fails(valid_transactions):
    valid_transactions.drop(columns=["amount"], inplace=True)
    
    with pytest.raises(ValueError, match = "Missing required columns"):
        validate_transactions(valid_transactions)                
    
            