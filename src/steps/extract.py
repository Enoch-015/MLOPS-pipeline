"""
Data Extraction Step

Pure function to extract data from the PostgreSQL database (or generate synthetic data).
Returns a pandas DataFrame. No ClearML Task/Artifacts here.
"""

import logging
import pandas as pd
from sqlalchemy import create_engine
import os
import tempfile
import uuid
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import configuration
from config import DB_CONNECTION_STRING, ETL_CONFIG, LOG_CONFIG, LOGS_DIR

# Setup logging
log_file = LOGS_DIR / f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_data(max_records: Optional[int] = None) -> pd.DataFrame:
    """Extract data from PostgreSQL database (or generate synthetic data).

    Args:
        max_records: Max number of rows to load (default: 50000)

    Returns:
        pandas.DataFrame: Extracted data
    """
    logger.info(f"extract_data called with max_records={max_records}")

    if max_records is None:
        max_records = 50000

    logger.info("Connecting to database...")
    df = None

    try:
        engine = create_engine(DB_CONNECTION_STRING)

        logger.info(f"Loading first {max_records} records from {ETL_CONFIG['table_name']} table...")

        # SQL query to load data with limit
        query = f"""
        SELECT * FROM {ETL_CONFIG['table_name']} 
        ORDER BY etl_processed_at, etl_batch_id
        LIMIT {max_records}
        """

        # Load data into DataFrame
        df = pd.read_sql_query(query, engine)
    except Exception as e:
        logger.warning(f"Failed to connect to database: {str(e)}")
        logger.info("Using synthetic data instead...")

        # Generate synthetic data for testing
        import numpy as np

        # Create a synthetic dataset with similar structure
        num_records = min(max_records, 1000)  # Limit to 1000 records for testing

        # Generate synthetic dates
        from datetime import datetime, timedelta
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(num_records)]

        # Create synthetic dataframe
        df = pd.DataFrame({
            'Order_ID': [f'ORD-{i:06d}' for i in range(num_records)],
            'Customer_ID': [f'CUST-{i % 100:04d}' for i in range(num_records)],
            'Customer_Type': np.random.choice(['Regular', 'Premium', 'New'], num_records),
            'Product': np.random.choice(['Coffee', 'Tea', 'Soda', 'Water', 'Juice'], num_records),
            'Category': np.random.choice(['Hot', 'Cold', 'Bottled'], num_records),
            'Unit_Price': np.random.uniform(1.0, 10.0, num_records).round(2),
            'Quantity': np.random.randint(1, 10, num_records),
            'Discount': np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2], num_records),
            'Total_Price': np.random.uniform(5.0, 50.0, num_records).round(2),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], num_records),
            'Order_Date': dates,
            'etl_processed_at': datetime.now(),
            'etl_batch_id': 1
        })

        logger.info("Generated synthetic data successfully")

    # Log data info
    logger.info(f"Successfully loaded {len(df)} records")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df

if __name__ == "__main__":
    # Run locally for quick check
    df = extract_data()
    print(df.head())
