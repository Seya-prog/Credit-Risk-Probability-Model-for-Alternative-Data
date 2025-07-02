"""Unit tests for data processing functions."""

import pytest
import pandas as pd
import numpy as np
from src.model_training import load_and_split_data
from src.proxy_target_engineering import calculate_rfm_metrics, identify_high_risk_cluster

def test_load_and_split_data(tmp_path):
    """Test data loading and splitting functionality."""
    # Create a sample dataset
    df = pd.DataFrame({
        'CustomerId': range(100),
        'TransactionYear': [2019] * 100,
        'TransactionMonth': [1] * 100,
        'TransactionDay': [1] * 100,
        'TransactionHour': [12] * 100,
        'TotalTransactionAmount': np.random.random(100),
        'is_high_risk': np.random.choice([0, 1], size=100)
    })
    
    # Save to temporary file
    data_path = tmp_path / "test_data.csv"
    df.to_csv(data_path, index=False)
    
    # Test splitting
    X_train, X_test, y_train, y_test = load_and_split_data(
        data_path=str(data_path),
        test_size=0.2,
        random_state=42
    )
    
    # Check shapes
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    # Check that CustomerId is not in features
    assert 'CustomerId' not in X_train.columns
    assert 'is_high_risk' not in X_train.columns

def test_calculate_rfm_metrics():
    """Test RFM metrics calculation."""
    # Create sample transaction data
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'TransactionYear': [2019, 2019, 2019],
        'TransactionMonth': [1, 2, 1],
        'TransactionDay': [1, 1, 1],
        'TransactionHour': [12, 12, 12],
        'TotalTransactionAmount': [100, 200, 150]
    })
    
    # Calculate RFM metrics
    rfm_df = calculate_rfm_metrics(df)
    
    # Check output structure
    assert isinstance(rfm_df, pd.DataFrame)
    assert all(col in rfm_df.columns for col in ['CustomerId', 'Recency', 'Frequency', 'MonetaryTotal'])
    
    # Check calculations
    assert len(rfm_df) == 2  # Unique customers
    assert rfm_df.loc[rfm_df['CustomerId'] == 1, 'Frequency'].iloc[0] == 2
    assert rfm_df.loc[rfm_df['CustomerId'] == 2, 'Frequency'].iloc[0] == 1

def test_identify_high_risk_cluster():
    """Test high-risk cluster identification."""
    # Create sample clustered data
    df = pd.DataFrame({
        'CustomerId': range(100),
        'Recency': np.random.randint(1, 100, 100),
        'Frequency': np.random.randint(1, 10, 100),
        'MonetaryTotal': np.random.random(100) * 1000,
        'Cluster': np.random.choice([0, 1, 2], size=100)
    })
    
    # Identify high-risk cluster
    high_risk_cluster = identify_high_risk_cluster(df)
    
    # Check output
    assert isinstance(high_risk_cluster, int)
    assert high_risk_cluster in [0, 1, 2] 