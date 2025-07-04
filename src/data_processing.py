"""
Data processing module for credit risk modeling.

This module implements a complete data processing pipeline that transforms
raw transaction data into model-ready format using sklearn.pipeline.Pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
import logging
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define feature column types
NUMERIC_FEATURES = ['TotalTransactionAmount', 'AverageTransactionAmount', 
                   'TransactionAmountStd', 'TransactionCount']

CATEGORICAL_FEATURES = ['TransactionHour', 'TransactionDay', 'TransactionMonth',
                       'TransactionYear', 'TransactionDayOfWeek']


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer for extracting datetime features."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Extracting datetime features")
        X_copy = X.copy()
        
        if 'TransactionStartTime' in X_copy.columns:
            if not pd.api.types.is_datetime64_dtype(X_copy['TransactionStartTime']):
                X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
            
            # Extract time-based features as per requirements
            X_copy['TransactionHour'] = X_copy['TransactionStartTime'].dt.hour
            X_copy['TransactionDay'] = X_copy['TransactionStartTime'].dt.day
            X_copy['TransactionMonth'] = X_copy['TransactionStartTime'].dt.month
            X_copy['TransactionYear'] = X_copy['TransactionStartTime'].dt.year
            X_copy['TransactionDayOfWeek'] = X_copy['TransactionStartTime'].dt.dayofweek
            
            # Drop original datetime column as it's been transformed
            X_copy.drop('TransactionStartTime', axis=1, inplace=True)
            
        return X_copy


class AggregateFeatureGenerator(BaseEstimator, TransformerMixin):
    """Custom transformer for generating aggregate features."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logger.info("Creating aggregate features")
        X_copy = X.copy()
        
        # Choose amount column (prefer Value over Amount if available)
        amount_col = 'Value' if 'Value' in X_copy.columns else 'Amount'
        
        if 'CustomerId' not in X_copy.columns or amount_col not in X_copy.columns:
            logger.warning("Required columns missing for aggregate features")
            return X_copy
        
        # Create aggregate features as per requirements
        agg_features = X_copy.groupby('CustomerId').agg({
            # Transaction count
            'TransactionId': 'count',
            
            # Amount-based features
            amount_col: [
                'sum',  # Total Transaction Amount
                'mean',  # Average Transaction Amount
                'std',  # Standard Deviation of Transaction Amounts
                'min',
                'max',
                'count'  # Transaction Count
            ]
        })
        
        # Flatten column names
        agg_features.columns = ['_'.join(col).strip('_') for col in agg_features.columns.values]
        
        # Rename columns for clarity
        rename_dict = {
            f'{amount_col}_sum': 'TotalTransactionAmount',
            f'{amount_col}_mean': 'AverageTransactionAmount',
            f'{amount_col}_std': 'TransactionAmountStd',
            'TransactionId_count': 'TransactionCount'
        }
        agg_features.rename(columns=rename_dict, inplace=True)
        
        # Fill NaN values in standard deviation with 0 (for customers with single transaction)
        if 'TransactionAmountStd' in agg_features.columns:
            # Fix: Avoid chained assignment with inplace=True
            agg_features = agg_features.fillna({'TransactionAmountStd': 0})
        
        return agg_features.reset_index()


def build_preprocessing_pipeline() -> Pipeline:
    """
    Build the complete preprocessing pipeline using sklearn.pipeline.Pipeline.
    
    Returns:
        Pipeline object with all transformation steps
    """
    # Create preprocessing steps for different column types
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Create the full pipeline - order matters here!
    # First extract datetime features, then generate aggregates, then preprocess
    full_pipeline = Pipeline([
        ('datetime_features', DateTimeFeatureExtractor()),
        ('aggregate_features', AggregateFeatureGenerator()),
        # Apply column transformers after the features are created
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, NUMERIC_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ]))
    ])
    
    return full_pipeline


def process_data(raw_data_path: str, output_dir: str = './data/processed') -> pd.DataFrame:
    """
    Process raw transaction data into model-ready format.
    
    Args:
        raw_data_path: Path to raw data file
        output_dir: Directory to save processed data
        
    Returns:
        Processed DataFrame ready for modeling
    """
    logger.info(f"Loading data from {raw_data_path}")
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    
    # Apply transformers manually in sequence
    # 1. Extract datetime features
    datetime_extractor = DateTimeFeatureExtractor()
    df_with_datetime = datetime_extractor.fit_transform(df)
    
    # 2. Generate aggregate features
    agg_generator = AggregateFeatureGenerator()
    df_agg = agg_generator.fit_transform(df_with_datetime)
    
    # 3. Process numeric features
    if all(feature in df_agg.columns for feature in NUMERIC_FEATURES):
        # Apply numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Extract numeric features
        numeric_data = df_agg[NUMERIC_FEATURES].copy()
        
        # Transform numeric data
        numeric_processed = numeric_transformer.fit_transform(numeric_data)
        
        # Create DataFrame with processed numeric data
        numeric_df = pd.DataFrame(
            data=numeric_processed,
            columns=pd.Index(NUMERIC_FEATURES),
            index=df_agg.index
        )
    else:
        logger.warning("Not all numeric features are available")
        numeric_df = pd.DataFrame(index=df_agg.index)
    
    # 4. Process categorical features if they exist
    if all(feature in df_agg.columns for feature in CATEGORICAL_FEATURES):
        # Apply categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Extract categorical features
        categorical_data = df_agg[CATEGORICAL_FEATURES].copy()
        
        # Transform categorical data
        categorical_processed = categorical_transformer.fit_transform(categorical_data)
        
        # Get feature names from one-hot encoder
        onehot_encoder = categorical_transformer.named_steps['onehot']
        
        # Generate feature names for one-hot encoded features
        cat_feature_names = []
        for i, feature in enumerate(CATEGORICAL_FEATURES):
            if i < len(onehot_encoder.categories_):
                # Skip first category due to drop='first'
                categories = list(onehot_encoder.categories_[i])[1:]
                for category in categories:
                    cat_feature_names.append(f"{feature}_{category}")
        
        # Create DataFrame with processed categorical data
        categorical_df = pd.DataFrame(
            data=categorical_processed,
            columns=pd.Index(cat_feature_names),
            index=df_agg.index
        )
    else:
        logger.warning("Not all categorical features are available")
        categorical_df = pd.DataFrame(index=df_agg.index)
    
    # 5. Combine all processed features
    # Start with customer ID if available
    if 'CustomerId' in df_agg.columns:
        result_df = pd.DataFrame({'CustomerId': df_agg['CustomerId']})
    else:
        result_df = pd.DataFrame(index=df_agg.index)
    
    # Extract and add datetime features for the first transaction of each customer
    # to preserve the customer-level aggregation
    if all(feature in df_with_datetime.columns for feature in CATEGORICAL_FEATURES):
        # Group by CustomerId and take the first transaction's datetime features
        datetime_features = df_with_datetime.groupby('CustomerId')[CATEGORICAL_FEATURES].first().reset_index()
        
        # Add datetime features to result
        if not datetime_features.empty and 'CustomerId' in datetime_features.columns:
            result_df = pd.merge(result_df, datetime_features, on='CustomerId', how='left')
    
    # Add numeric and categorical features
    for other_df in [numeric_df, categorical_df]:
        if not other_df.empty:
            result_df = pd.concat([result_df, other_df], axis=1)
    
    # Create output directory and save processed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'processed_dataset.csv')
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved processed dataset to {output_path}")
    
    return result_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process raw transaction data')
    parser.add_argument('--input', type=str, default='./data/raw/data.csv',
                       help='Path to the input raw data file')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Directory to save processed data')
    
    args = parser.parse_args()
    process_data(args.input, args.output_dir) 