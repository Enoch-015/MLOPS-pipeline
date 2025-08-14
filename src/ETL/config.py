"""
Configuration file for ETL processes
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration - using environment variables
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'postgres'),
    'username': os.getenv('DB_USERNAME', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

# Construct connection string
DB_CONNECTION_STRING = f"postgresql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"

# File paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'src' / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
LOGS_DIR = PROJECT_ROOT / 'logs'

# ETL Configuration - using environment variables
ETL_CONFIG = {
    'chunk_size': int(os.getenv('ETL_CHUNK_SIZE', 10000)),
    'table_name': os.getenv('ETL_TABLE_NAME', 'beverage_sales'),
    'batch_size': int(os.getenv('ETL_BATCH_SIZE', 1000)),
    'max_retries': int(os.getenv('ETL_MAX_RETRIES', 3)),
    'retry_delay': int(os.getenv('ETL_RETRY_DELAY', 5)),  # seconds
}

# Data file paths
BEVERAGE_DATA_FILE = RAW_DATA_DIR / 'synthetic_beverage_sales_data.csv'

# Logging configuration - using environment variables
LOG_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'filename': 'etl.log'
}

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
