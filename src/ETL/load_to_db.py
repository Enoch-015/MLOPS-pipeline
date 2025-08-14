"""
ETL Pipeline for Beverage Sales Data
Loads data from CSV to PostgreSQL database
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
from pathlib import Path
import sys
from datetime import datetime
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration
from config import (
    DB_CONNECTION_STRING, 
    ETL_CONFIG, 
    LOG_CONFIG, 
    BEVERAGE_DATA_FILE
)

# Configure logging using config
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOG_CONFIG['filename']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BeverageETL:
    """ETL class for beverage sales data"""
    
    def __init__(self):
        self.engine = create_engine(DB_CONNECTION_STRING)
        self.data_path = BEVERAGE_DATA_FILE
        self.table_name = ETL_CONFIG['table_name']
        self.chunk_size = ETL_CONFIG['chunk_size']
        self.batch_size = ETL_CONFIG['batch_size']
        self.max_retries = ETL_CONFIG['max_retries']
        self.retry_delay = ETL_CONFIG['retry_delay']
        
    def extract_data(self):
        """Extract data from CSV file"""
        try:
            logger.info(f"Extracting data from {self.data_path}")
            
            # Check if file exists
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Read CSV in chunks to handle large file
            logger.info("Reading CSV file...")
            df_chunks = pd.read_csv(self.data_path, chunksize=self.chunk_size)
            
            logger.info("Data extraction completed successfully")
            return df_chunks
            
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            raise
    
    def transform_data(self, df_chunk):
        """Transform and clean data"""
        try:
            logger.debug(f"Transforming chunk with {len(df_chunk)} rows")
            
            # Create a copy to avoid modifying original
            df = df_chunk.copy()
            
            # Basic data cleaning and transformation
            # Convert date columns if they exist
            date_columns = df.select_dtypes(include=['object']).columns
            for col in date_columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
            
            # Clean numeric columns
            numeric_columns = df.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass
            
            # Remove duplicates if any
            initial_rows = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_rows:
                logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
            
            # Handle missing values
            # Fill numeric NaNs with 0 or median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical NaNs with 'Unknown'
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna('Unknown')
            
            # Add metadata columns
            df['etl_processed_at'] = datetime.now()
            df['etl_batch_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            logger.debug(f"Transformation completed. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def create_table(self, sample_df):
        """Create table based on DataFrame schema"""
        try:
            logger.info(f"Creating table: {self.table_name}")
            
            # Check if table already exists
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = '{self.table_name}'
                    );
                """))
                table_exists = result.fetchone()[0]
                
                if table_exists:
                    logger.info(f"Table {self.table_name} already exists. Dropping and recreating...")
                    conn.execute(text(f"DROP TABLE IF EXISTS {self.table_name}"))
                    conn.commit()
            
            # Create table using pandas (it will infer data types)
            sample_df.head(0).to_sql(
                self.table_name, 
                self.engine, 
                if_exists='replace', 
                index=False,
                method='multi'
            )
            
            logger.info(f"Table {self.table_name} created successfully")
            
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            raise
    
    def load_data(self, df):
        """Load data into PostgreSQL database"""
        try:
            logger.debug(f"Loading chunk with {len(df)} rows to database")
            
            # Load data to PostgreSQL
            df.to_sql(
                self.table_name,
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.debug(f"Successfully loaded {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def verify_load(self):
        """Verify that data was loaded correctly"""
        try:
            logger.info("Verifying data load...")
            
            with self.engine.connect() as conn:
                # Count total rows
                result = conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}"))
                total_rows = result.fetchone()[0]
                logger.info(f"Total rows in database: {total_rows:,}")
                
                # Get sample data
                result = conn.execute(text(f"SELECT * FROM {self.table_name} LIMIT 5"))
                sample_data = result.fetchall()
                
                logger.info("Sample data from database:")
                for row in sample_data[:3]:  # Show first 3 rows
                    logger.info(f"  {row}")
                
                # Get column info
                result = conn.execute(text(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{self.table_name}'
                    ORDER BY ordinal_position
                """))
                columns = result.fetchall()
                
                logger.info(f"Table schema ({len(columns)} columns):")
                for col_name, col_type in columns[:10]:  # Show first 10 columns
                    logger.info(f"  {col_name}: {col_type}")
                
                return total_rows
                
        except Exception as e:
            logger.error(f"Error verifying data: {str(e)}")
            raise
    
    def run_etl(self):
        """Run the complete ETL process"""
        try:
            logger.info("=== Starting ETL Process ===")
            start_time = datetime.now()
            
            # Extract data
            df_chunks = self.extract_data()
            
            # Process first chunk to create table
            logger.info("Processing first chunk to create table schema...")
            first_chunk = next(df_chunks)
            transformed_first = self.transform_data(first_chunk)
            
            # Create table
            self.create_table(transformed_first)
            
            # Load first chunk
            self.load_data(transformed_first)
            total_processed = len(transformed_first)
            logger.info(f"Processed chunk 1: {len(transformed_first):,} rows")
            
            # Process remaining chunks
            chunk_num = 2
            for chunk in df_chunks:
                logger.info(f"Processing chunk {chunk_num}...")
                transformed_chunk = self.transform_data(chunk)
                self.load_data(transformed_chunk)
                total_processed += len(transformed_chunk)
                logger.info(f"Processed chunk {chunk_num}: {len(transformed_chunk):,} rows (Total: {total_processed:,})")
                chunk_num += 1
            
            # Verify the load
            db_rows = self.verify_load()
            
            # Log completion
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("=== ETL Process Completed Successfully ===")
            logger.info(f"Total rows processed: {total_processed:,}")
            logger.info(f"Total rows in database: {db_rows:,}")
            logger.info(f"Total duration: {duration}")
            logger.info(f"Average processing speed: {total_processed/duration.total_seconds():.0f} rows/second")
            
            return True
            
        except Exception as e:
            logger.error(f"ETL process failed: {str(e)}")
            raise

def main():
    """Main function to run ETL"""
    try:
        etl = BeverageETL()
        etl.run_etl()
        print("✅ ETL process completed successfully!")
        
    except Exception as e:
        print(f"❌ ETL process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
