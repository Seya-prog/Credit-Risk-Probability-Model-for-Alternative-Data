"""Unit tests for data processing functions."""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_and_split_data
from src.proxy_target_engineering import calculate_rfm_metrics, identify_high_risk_cluster

def test_load_and_split_data():
    """Test the data loading and splitting functionality"""
    # Create a mock dataset
    mock_data = pd.DataFrame({
        'CustomerId': range(100),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'is_high_risk': np.random.choice([0, 1], size=100)
    })
    
    # Save mock data
    mock_data.to_csv('./data/processed/test_dataset.csv', index=False)
    
    # Test the function
    X_train, X_test, y_train, y_test = load_and_split_data(
        data_path='./data/processed/test_dataset.csv',
        test_size=0.2,
        random_state=42
    )
    
    # Verify shapes
    assert len(X_train) == 80  # 80% of 100
    assert len(X_test) == 20   # 20% of 100
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    # Verify types
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    
    # Verify no data leakage
    assert 'CustomerId' not in X_train
    assert 'is_high_risk' not in X_train

def test_data_preprocessing():
    """Test data preprocessing steps"""
    # Create mock data with missing values and mixed types
    mock_data = pd.DataFrame({
        'CustomerId': range(100),
        'numeric_feature': np.random.randn(100),
        'integer_feature': np.random.randint(0, 100, 100),
        'is_high_risk': np.random.choice([0, 1], size=100)
    })
    
    # Add some missing values
    mock_data.loc[0:10, 'numeric_feature'] = np.nan
    mock_data.to_csv('./data/processed/test_dataset.csv', index=False)
    
    # Test loading and preprocessing
    X_train, X_test, y_train, y_test = load_and_split_data(
        data_path='./data/processed/test_dataset.csv'
    )
    
    # Verify no missing values
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    
    # Verify all features are float64
    assert X_train.dtype == np.float64
    assert X_test.dtype == np.float64

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