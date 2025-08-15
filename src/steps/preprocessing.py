"""
Data Preprocessing Step

Pure function that preprocesses a DataFrame and returns a processed DataFrame
plus metadata (target column name and fitted label encoders). No ClearML code.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import configuration
from config import LOG_CONFIG, LOGS_DIR, DATA_SCHEMA

# Setup logging
log_file = LOGS_DIR / f"preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame):
    """Preprocess the data for training.

    Args:
        df: Raw dataframe.

    Returns:
        tuple[pd.DataFrame, dict]: (processed_df, metadata)
        where metadata contains:
        - 'target_column': str
        - 'label_encoders': dict
        - 'original_columns': list
        - 'processed_columns': list
    """
    logger.info(f"Loaded data with shape: {df.shape}")
    original_columns = list(df.columns)

    # Create a copy to avoid modifying original data
    df_processed = df.copy()

    # Drop ETL metadata columns and non-predictive columns
    columns_to_drop = ['etl_processed_at', 'etl_batch_id', 'Order_ID', 'Customer_ID']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]

    if existing_columns_to_drop:
        df_processed = df_processed.drop(columns=existing_columns_to_drop)
        logger.info(f"Dropped columns: {existing_columns_to_drop}")

    # Handle datetime columns (extract useful features)
    date_columns = DATA_SCHEMA['date_columns']
    for col in date_columns:
        if col in df_processed.columns:
            logger.info(f"Processing datetime column: {col}")
            # Convert to datetime if needed
            if df_processed[col].dtype != 'datetime64[ns]':
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')

            # Extract datetime features
            df_processed[f'{col}_year'] = df_processed[col].dt.year
            df_processed[f'{col}_month'] = df_processed[col].dt.month
            df_processed[f'{col}_day'] = df_processed[col].dt.day
            df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
            df_processed = df_processed.drop(columns=[col])  # Drop original datetime column

    # Check if target column exists
    target_candidates = DATA_SCHEMA['target_column_candidates']
    target_col = None

    for candidate in target_candidates:
        if candidate in df_processed.columns:
            target_col = candidate
            break

    if target_col is None:
        # Try to find any column with 'total' and 'price'
        target_candidates = [col for col in df_processed.columns if 'total' in col.lower() and 'price' in col.lower()]
        if target_candidates:
            target_col = target_candidates[0]
            logger.info(f"Using '{target_col}' as target variable")
        else:
            raise ValueError("Target column containing 'total' and 'price' not found in data")
    else:
        logger.info(f"Using '{target_col}' as target variable")

    # Handle missing values
    logger.info("Handling missing values...")
    df_processed = df_processed.dropna()
    logger.info(f"Data shape after removing NaN values: {df_processed.shape}")

    # Encode categorical variables
    logger.info("Encoding categorical variables...")
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        if col != target_col:  # Don't encode target if it's categorical
            logger.info(f"Encoding column: {col}")
            encoder = LabelEncoder()
            df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = encoder

    logger.info(f"Encoded {len(categorical_cols)} categorical columns")
    logger.info(f"Final data shape: {df_processed.shape}")

    # Create metadata dictionary with all preprocessing information
    metadata = {
        'target_column': str(target_col),  # Ensure it's explicitly a string
        'label_encoders': label_encoders,
        'original_columns': original_columns,
        'processed_columns': list(df_processed.columns),
        'columns_dropped': existing_columns_to_drop,
        'categorical_columns_encoded': list(label_encoders.keys()),
        'preprocessing_timestamp': datetime.now().isoformat(),
        'data_shape_before': df.shape,
        'data_shape_after': df_processed.shape
    }

    logger.info(f"Metadata created: target_column='{metadata['target_column']}' (type: {type(metadata['target_column'])})")

    return df_processed, metadata

if __name__ == "__main__":
    print("This module provides preprocess_data(df) pure function.")
    print("Returns: (processed_df, metadata_dict)")