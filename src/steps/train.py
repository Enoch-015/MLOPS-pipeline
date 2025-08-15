"""
Model Training Step

Pure function to train an XGBoost model on preprocessed data.
Returns the trained model and metrics. No ClearML Task code here.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
import os
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import configuration
from config import LOG_CONFIG, LOGS_DIR, MODEL_HYPERPARAMS, MODELS_DIR

# Setup logging
log_file = LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _normalize_target_col(value, columns):
    """Normalize target_col value to an existing column name.

    Handles cases where value may be wrapped by the pipeline as nested lists/tuples
    or split into a tuple/list of characters.
    """
    # Fast path
    if isinstance(value, str) and value in columns:
        return value

    # If single-element list/tuple, unwrap recursively
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _normalize_target_col(value[0], columns)

    # If it's a list/tuple of single-character strings, join them (e.g., ('T','o',...))
    if isinstance(value, (list, tuple)) and value and all(isinstance(c, str) and len(c) == 1 for c in value):
        joined = "".join(value)
        if joined in columns:
            return joined
        # case-insensitive fallback
        for col in columns:
            if col.lower() == joined.lower():
                return col
        return joined

    # If something like [ ('T','o',...) ]
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return _normalize_target_col(value[0], columns)

    # Case-insensitive direct match (handle numpy scalars)
    try:
        import numpy as _np  # local import to avoid polluting namespace
        if isinstance(value, (_np.generic,)):
            val_str = str(value.item())
        else:
            val_str = str(value)
    except Exception:
        val_str = str(value)
    for col in columns:
        if col.lower() == val_str.lower():
            return col
    return val_str

# Update your train.py train_model function:

def train_model(df_processed: pd.DataFrame, target_col: str):
    """Train XGBoost model on preprocessed data."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Robust target column validation and normalization
    logger.info(f"train_model received target_col: {repr(target_col)} (type: {type(target_col)})")
    
    # Ensure target_col is a proper string
    if isinstance(target_col, (list, tuple)):
        if len(target_col) == 1:
            target_col = str(target_col[0])
        elif all(isinstance(c, str) and len(c) == 1 for c in target_col):
            target_col = ''.join(target_col)
        else:
            target_col = str(target_col)
    target_col = str(target_col).strip()
    
    # Verify target column exists
    if target_col not in df_processed.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame columns: {list(df_processed.columns)}")
        # Try case-insensitive match as fallback
        for col in df_processed.columns:
            if col.lower() == target_col.lower():
                target_col = col
                logger.info(f"Found case-insensitive match: {target_col}")
                break
        else:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    
    logger.info(f"Using target column: '{target_col}'")
    
    # Separate features and target - use the robust drop method
    try:
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
    except KeyError as e:
        logger.error(f"Failed to drop target column: {e}")
        logger.error(f"Available columns: {list(df_processed.columns)}")
        raise
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # Rest of your training code remains the same...
    # (continue with your existing training logic)
    logger.info(f"Feature columns: {X.columns.tolist()}")

    # Split data
    train_ratio = MODEL_HYPERPARAMS.get('train_test_split_ratio', 0.5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=MODEL_HYPERPARAMS['random_state'], train_size=train_ratio
    )

    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")

    # Train XGBoost model with hyperparameters from config
    logger.info("Training XGBoost model...")
    xgb_params = {k: v for k, v in MODEL_HYPERPARAMS.items() if k not in ['train_test_split_ratio']}

    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train)

    # Evaluate model
    train_score = xgb.score(X_train, y_train)
    test_score = xgb.score(X_test, y_test)

    logger.info("Model training completed")
    logger.info(f"Training R² Score: {train_score:.4f}")
    logger.info(f"Test R² Score: {test_score:.4f}")

    # Save model performance metrics
    metrics = {
        'train_score': float(train_score),
        'test_score': float(test_score),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'n_features': int(X_train.shape[1]),
        'training_date': datetime.now().isoformat()
    }

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Persist model locally for convenience
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_model_{timestamp}"
    local_model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(local_model_path, 'wb') as f:
        pickle.dump(xgb, f)
    logger.info(f"Model saved locally to: {local_model_path}")

    latest_info = {
        'model_id': None,
        'model_path': str(local_model_path),
        'timestamp': timestamp,
        'metrics': metrics,
    }
    latest_path = MODELS_DIR / "latest_model_info.pkl"
    with open(latest_path, 'wb') as f:
        pickle.dump(latest_info, f)
    logger.info(f"Latest model info saved to: {latest_path}")

    return xgb, metrics, feature_importance

if __name__ == "__main__":
    print("This module provides train_model(df_processed, target_col) pure function.")
