"""
Model Training and Evaluation Module.

This module implements the training pipeline with MLflow tracking:
- Data splitting
- Model training (multiple algorithms)
- Hyperparameter tuning
- Model evaluation
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import Model
from mlflow.tracking import MlflowClient
import logging
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_split_data(
    data_path: str = './data/processed/processed_dataset.csv',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split data into training and testing sets.
    
    Args:
        data_path: Path to the processed dataset
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Loading and splitting data")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = df['is_high_risk'].astype('float64')  # Convert target to float64
    
    # Convert all integer columns to float64 to handle potential missing values
    int_columns = X.select_dtypes(include=['int64', 'int32']).columns
    X[int_columns] = X[int_columns].astype('float64')
    logger.info(f"Converted {len(int_columns)} integer columns to float64")
    
    # Split data
    X_train, X_test, y_train, y_test = map(np.array, train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    ))
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    params: Dict[str, Any] | None = None
) -> Dict[str, float]:
    """
    Train and evaluate a model with MLflow tracking.
    """
    logger.info(f"Training and evaluating {model_name}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log parameters and metrics
    if params:
        mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    # Save model directly with MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name + "_model",
        input_example=X_train[:5]
    )
    
    logger.info(f"Model {model_name} metrics: {metrics}")
    return metrics

def grid_search_cv(
    model: Any,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    cv: int = 5
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform grid search cross-validation.
    
    Args:
        model: Base model instance
        param_grid: Parameter grid for search
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (best_model, best_params)
    """
    logger.info(f"Performing grid search for {model_name}")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='roc_auc',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def manually_create_latest_run(model, X_train, mlruns_dir="./mlruns"):
    """
    Manually create a run with ID 'latest' by directly manipulating the MLflow directory structure.
    
    Args:
        model: Trained model to save
        X_train: Training data (for example input)
        mlruns_dir: Path to MLflow runs directory
    """
    logger.info("Manually creating 'latest' run for the API")
    
    # Create necessary directories
    latest_dir = os.path.join(mlruns_dir, "0", "latest")
    artifacts_dir = os.path.join(latest_dir, "artifacts")
    model_dir = os.path.join(artifacts_dir, "gradient_boosting_model")  # Fixed model name for API
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create meta.yaml file with relative path
    meta_content = """artifact_uri: file:./mlruns/0/latest/artifacts
end_time: 0
entry_point_name: ''
experiment_id: '0'
lifecycle_stage: active
run_id: latest
run_name: gradient_boosting
source_name: ''
source_type: 4
source_version: ''
start_time: 0
status: 1
tags: []
user_id: api_user
"""
    
    with open(os.path.join(latest_dir, "meta.yaml"), "w") as f:
        f.write(meta_content)
    
    # Save the model using MLflow for consistency
    mlflow.sklearn.save_model(
        sk_model=model,
        path=model_dir,
        input_example=X_train[:5]
    )
    
    # Verify artifacts were created
    existing_files = os.listdir(model_dir)
    logger.info(f"Created artifacts in {model_dir}:")
    for file in existing_files:
        logger.info(f"- {file}")
    
    # Verify model file size
    model_path = os.path.join(model_dir, "model.pkl")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model file size: {size_mb:.2f} MB")
    
    logger.info(f"Successfully created 'latest' run at {latest_dir}")

def main():
    """Main execution function."""
    # Set MLflow tracking URI (local)
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Ensure mlruns directory exists with proper structure
    os.makedirs("./mlruns/0", exist_ok=True)
    
    # Create default experiment metadata if it doesn't exist
    meta_path = "./mlruns/0/meta.yaml"
    if not os.path.exists(meta_path):
        meta_content = """artifact_location: ./mlruns/0
experiment_id: '0'
lifecycle_stage: active
name: Default"""
        with open(meta_path, "w") as f:
            f.write(meta_content)
    
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Define models and their parameter grids
    models = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    
    # Train and evaluate each model
    best_models = {}
    for name, config in models.items():
        try:
            logger.info(f"Training {name}")
            
            # Perform grid search
            best_model, best_params = grid_search_cv(
                config['model'],
                config['params'],
                X_train,
                y_train,
                name
            )
            
            # Create a new run for each model
            with mlflow.start_run(run_name=name):
                metrics = train_and_evaluate_model(
                    best_model,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    name,
                    best_params
                )
                
                best_models[name] = {
                    'model': best_model,
                    'metrics': metrics
                }
        
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            # Ensure any active run is ended
            mlflow.end_run()
            continue
    
    # Ensure any remaining run is ended
    mlflow.end_run()
    
    if not best_models:
        logger.error("No models were successfully trained")
        return
    
    # Identify best model
    best_model_name = max(
        best_models.keys(),
        key=lambda k: best_models[k]['metrics']['roc_auc']
    )
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best model metrics: {best_models[best_model_name]['metrics']}")
    
    # Create the special 'latest' run for the gradient boosting model
    gradient_boosting_model = best_models.get('gradient_boosting', {}).get('model')
    if gradient_boosting_model:
        manually_create_latest_run(gradient_boosting_model, X_train)
    else:
        logger.error("Gradient boosting model not found, cannot create latest run")

if __name__ == "__main__":
    main() 