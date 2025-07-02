"""
Proxy Target Variable Engineering for Credit Risk Modeling.

This module creates a proxy 'credit risk' column by identifying disengaged customers
using RFM (Recency, Frequency, Monetary) analysis and K-means clustering.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Tuple, Dict, Any, cast, List
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from numpy.typing import NDArray

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_rfm_metrics(transactions_df: DataFrame, snapshot_date=None) -> DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    
    Args:
        transactions_df: DataFrame containing transaction data
        snapshot_date: Reference date for recency calculation (default: max date in data)
        
    Returns:
        DataFrame with CustomerID and RFM metrics
    """
    logger.info("Calculating RFM metrics")
    
    # Make a copy to avoid modifying the original
    df = transactions_df.copy()
    
    # Create transaction datetime from components
    date_components = df[['TransactionYear', 'TransactionMonth', 'TransactionDay', 'TransactionHour']].copy()
    date_components.columns = ['year', 'month', 'day', 'hour']
    date_components['minute'] = 0
    date_components['second'] = 0
    df['TransactionDate'] = pd.to_datetime(date_components)
    
    # Set snapshot date if not provided
    if snapshot_date is None:
        snapshot_date = df['TransactionDate'].max()
    elif isinstance(snapshot_date, str):
        snapshot_date = pd.to_datetime(snapshot_date)
    
    logger.info(f"Using snapshot date: {snapshot_date}")
    
    # Calculate RFM metrics
    rfm = df.groupby('CustomerId').agg({
        'TransactionDate': lambda x: (snapshot_date - x.max()).days,  # Recency (days since last transaction)
        'CustomerId': 'count',  # Frequency (number of transactions)
        'TotalTransactionAmount': ['sum', 'mean']  # Monetary (total & average spending)
    })
    
    # Flatten the column names
    rfm.columns = pd.Index([str('_'.join(col)).strip('_') for col in rfm.columns.values])
    
    # Create a new DataFrame with renamed columns
    rfm_renamed = DataFrame(
        data=rfm.values,
        columns=pd.Index([
            'Recency',
            'Frequency',
            'MonetaryTotal',
            'MonetaryAverage'
        ]),
        index=rfm.index
    )
    
    # Reset index to make CustomerId a column
    rfm_renamed = rfm_renamed.reset_index()
    
    # Debug: Print column names
    logger.info(f"RFM DataFrame columns: {rfm_renamed.columns.tolist()}")
    
    logger.info(f"Generated RFM metrics for {len(rfm_renamed)} customers")
    return rfm_renamed

def cluster_customers(rfm_df: DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[DataFrame, KMeans]:
    """
    Cluster customers based on their RFM metrics using K-means.
    
    Args:
        rfm_df: DataFrame with RFM metrics
        n_clusters: Number of clusters to create
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (DataFrame with cluster assignments, KMeans model)
    """
    logger.info(f"Clustering customers into {n_clusters} segments")
    
    # Select RFM features for clustering
    rfm_features = ['Recency', 'Frequency', 'MonetaryTotal']
    
    # Validate required columns
    if not all(feature in rfm_df.columns for feature in rfm_features):
        missing = [feat for feat in rfm_features if feat not in rfm_df.columns]
        logger.error(f"Missing required features: {missing}")
        raise ValueError(f"Missing required features: {missing}")
    
    # Extract features for clustering
    features = rfm_df[rfm_features].copy()
    
    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    cluster_labels = kmeans.fit_predict(features_scaled)
    rfm_df_copy = rfm_df.copy()
    rfm_df_copy.loc[:, 'Cluster'] = cluster_labels
    
    # Calculate cluster centers for interpretation
    centers_df = DataFrame(
        data=scaler.inverse_transform(kmeans.cluster_centers_),
        columns=pd.Index(rfm_features),
        index=pd.RangeIndex(n_clusters)
    )
    
    logger.info(f"Cluster centers:\n{centers_df}")
    return rfm_df_copy, kmeans

def identify_high_risk_cluster(clustered_df: DataFrame) -> int:
    """
    Identify which cluster represents the high-risk customers (disengaged).
    
    Args:
        clustered_df: DataFrame with cluster assignments
        
    Returns:
        Cluster ID representing high-risk customers
    """
    logger.info("Identifying high-risk cluster")
    
    # Calculate average RFM values for each cluster
    cluster_stats = clustered_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryTotal': 'mean'
    }).reset_index()
    
    # Define the high-risk cluster as having:
    # 1. High Recency (longest time since last transaction)
    # 2. Low Frequency (fewest transactions)
    # 3. Low MonetaryTotal (least amount spent)
    
    # Normalize scores for each metric from 0-1 where 1 is highest risk
    cluster_stats.loc[:, 'RecencyScore'] = cluster_stats['Recency'] / cluster_stats['Recency'].max()
    cluster_stats.loc[:, 'FrequencyScore'] = 1 - (cluster_stats['Frequency'] / cluster_stats['Frequency'].max())
    cluster_stats.loc[:, 'MonetaryScore'] = 1 - (cluster_stats['MonetaryTotal'] / cluster_stats['MonetaryTotal'].max())
    
    # Calculate total risk score
    cluster_stats.loc[:, 'RiskScore'] = (
        cluster_stats['RecencyScore'] +
        cluster_stats['FrequencyScore'] +
        cluster_stats['MonetaryScore']
    ) / 3
    
    # Identify the cluster with the highest risk score
    high_risk_cluster = int(cluster_stats.loc[cluster_stats['RiskScore'].idxmax(), 'Cluster'])
    
    logger.info(f"High risk cluster identified: {high_risk_cluster}")
    logger.info(f"Cluster statistics:\n{cluster_stats}")
    
    return high_risk_cluster

def create_risk_labels(clustered_df: DataFrame, high_risk_cluster: int) -> DataFrame:
    """
    Create binary risk labels based on cluster assignments.
    
    Args:
        clustered_df: DataFrame with cluster assignments
        high_risk_cluster: Cluster ID representing high-risk customers
        
    Returns:
        DataFrame with binary risk labels
    """
    logger.info(f"Creating risk labels (high risk cluster: {high_risk_cluster})")
    
    # Create a copy to avoid modifying the original
    result_df = clustered_df.copy()
    
    # Create binary risk labels
    result_df.loc[:, 'is_high_risk'] = (result_df['Cluster'] == high_risk_cluster).astype(int)
    
    # Log statistics about the distribution
    risk_distribution = result_df['is_high_risk'].value_counts(normalize=True) * 100
    logger.info(f"Risk distribution (%):\n{risk_distribution}")
    
    return result_df

def generate_proxy_target(processed_data_path: str = './data/processed/processed_dataset.csv') -> DataFrame:
    """
    Generate proxy target variable (is_high_risk) using RFM analysis and clustering.
    Only outputs the CustomerId and is_high_risk columns.
    
    Args:
        processed_data_path: Path to the processed dataset
        
    Returns:
        DataFrame with CustomerId and is_high_risk columns
    """
    logger.info("Generating proxy target variable")
    
    # Read processed data
    df = pd.read_csv(processed_data_path)
    
    # Calculate RFM metrics
    rfm_df = calculate_rfm_metrics(df)
    
    # Cluster customers
    clustered_df, _ = cluster_customers(rfm_df)
    
    # Identify high-risk cluster
    high_risk_cluster = identify_high_risk_cluster(clustered_df)
    
    # Create risk labels
    risk_df = create_risk_labels(clustered_df, high_risk_cluster)
    
    # Return only CustomerId and is_high_risk columns
    result = risk_df[['CustomerId', 'is_high_risk']]
    return cast(DataFrame, result)

def main():
    """Main execution function."""
    try:
        # Generate proxy target
        risk_labels = generate_proxy_target()
        logger.info(f"Generated risk labels with columns: {risk_labels.columns.tolist()}")
        
        # Read the processed dataset
        processed_data_path = './data/processed/processed_dataset.csv'
        processed_data = pd.read_csv(processed_data_path)
        logger.info(f"Read processed data with columns: {processed_data.columns.tolist()}")
        
        # Drop existing is_high_risk column if it exists
        if 'is_high_risk' in processed_data.columns:
            processed_data = processed_data.drop('is_high_risk', axis=1)
            
        # Merge the risk labels with the processed dataset
        merged_data = processed_data.merge(risk_labels, on='CustomerId', how='left')
        logger.info(f"Merged data columns: {merged_data.columns.tolist()}")
        
        # Fill any missing values with 0 (non-risky) and convert to int
        merged_data['is_high_risk'] = merged_data['is_high_risk'].fillna(0).astype(int)
        
        # Save back to the same file
        merged_data.to_csv(processed_data_path, index=False)
        logger.info(f"Updated processed dataset with risk labels at {processed_data_path}")
        
    except Exception as e:
        logger.error(f"Error in proxy target generation: {str(e)}")
        logger.error(f"Risk labels shape: {risk_labels.shape if 'risk_labels' in locals() else 'not generated'}")
        raise

if __name__ == "__main__":
    main() 