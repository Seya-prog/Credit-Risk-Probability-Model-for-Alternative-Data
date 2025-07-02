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
import logging
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    params: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Train and evaluate a model with MLflow tracking.
    
    Args:
        model: Sklearn model instance
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name: Name of the model for logging
        params: Model parameters for logging
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Training and evaluating {model_name}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        if params:
            mlflow.log_params(params)
        
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
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Create model signature with explicit types
        input_schema = mlflow.types.Schema([
            mlflow.types.ColSpec(type=mlflow.types.DataType.double, name=col)
            for col in X_train.columns
        ])
        output_schema = mlflow.types.Schema([
            mlflow.types.ColSpec(type=mlflow.types.DataType.double, name="prediction")
        ])
        signature = mlflow.models.ModelSignature(
            inputs=input_schema,
            outputs=output_schema
        )
        
        # Log model with signature and input example
        mlflow.sklearn.log_model(
            model,
            name=f"{model_name}_model",
            signature=signature,
            input_example=X_train.iloc[:5]
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

def main():
    """Main execution function."""
    # Set MLflow tracking URI (local)
    mlflow.set_tracking_uri("file:./mlruns")
    
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
            
            # Evaluate best model
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
            continue
    
    # Identify best model
    best_model_name = max(
        best_models.keys(),
        key=lambda k: best_models[k]['metrics']['roc_auc']
    )
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best model metrics: {best_models[best_model_name]['metrics']}")

if __name__ == "__main__":
    main() 