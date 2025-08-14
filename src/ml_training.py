"""
Machine Learning Training Script for Beverage Sales Data
Loads data from PostgreSQL database and trains XGBoost model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sqlalchemy import create_engine, text
import logging
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'ETL'))

# Import configuration
from ETL.config import DB_CONNECTION_STRING, ETL_CONFIG, LOG_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BeverageMLTrainer:
    """ML training class for beverage sales prediction"""
    
    def __init__(self, max_records=1000000):
        """
        Initialize the trainer
        
        Args:
            max_records (int): Maximum number of records to load for training
        """
        self.engine = create_engine(DB_CONNECTION_STRING)
        self.table_name = ETL_CONFIG['table_name']
        self.max_records = max_records
        self.label_encoders = {}
        self.model = None
        self.model_path = Path(__file__).parent / 'models'
        self.model_path.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load data from PostgreSQL database"""
        try:
            logger.info(f"Loading first {self.max_records} records from {self.table_name} table...")
            
            # SQL query to load data with limit
            query = f"""
            SELECT * FROM {self.table_name} 
            ORDER BY etl_processed_at, etl_batch_id
            LIMIT {self.max_records}
            """
            
            # Load data into DataFrame
            df = pd.read_sql_query(query, self.engine)
            logger.info(f"Successfully loaded {len(df)} records")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        try:
            logger.info("Starting data preprocessing...")
            
            # Create a copy to avoid modifying original data
            df_processed = df.copy()
            
            # Drop ETL metadata columns and non-predictive columns
            columns_to_drop = ['etl_processed_at', 'etl_batch_id', 'Order_ID', 'Customer_ID']
            existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
            
            if existing_columns_to_drop:
                df_processed = df_processed.drop(columns=existing_columns_to_drop)
                logger.info(f"Dropped columns: {existing_columns_to_drop}")
            
            # Handle datetime columns (extract useful features)
            date_columns = df_processed.select_dtypes(include=['datetime64', 'datetime']).columns
            for col in date_columns:
                logger.info(f"Processing datetime column: {col}")
                df_processed[f'{col}_year'] = df_processed[col].dt.year
                df_processed[f'{col}_month'] = df_processed[col].dt.month
                df_processed[f'{col}_day'] = df_processed[col].dt.day
                df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
                df_processed = df_processed.drop(columns=[col])  # Drop original datetime column
            
            # Check if target column exists
            target_candidates = ['Total_Price', 'total_price', 'Total_price']
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
            
            for col in categorical_cols:
                if col != target_col:  # Don't encode target if it's categorical
                    logger.info(f"Encoding column: {col}")
                    encoder = LabelEncoder()
                    df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = encoder
            
            logger.info(f"Encoded {len(categorical_cols)} categorical columns")
            logger.info(f"Final data shape: {df_processed.shape}")
            
            return df_processed, target_col
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def train_model(self, df, target_col):
        """Train XGBoost model"""
        try:
            logger.info("Starting model training...")
            
            # Separate features and target
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Target shape: {y.shape}")
            logger.info(f"Feature columns: {X.columns.tolist()}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=19, train_size=0.50
            )
            
            logger.info(f"Training set size: {X_train.shape[0]}")
            logger.info(f"Test set size: {X_test.shape[0]}")
            
            # Train XGBoost model
            logger.info("Training XGBoost model...")
            xgb = XGBRegressor(n_estimators=200, random_state=42)
            self.model = xgb.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"✅ Model training completed!")
            logger.info(f"Training R² Score: {train_score:.4f}")
            logger.info(f"Test R² Score: {test_score:.4f}")
            
            # Save model performance metrics
            metrics = {
                'train_score': train_score,
                'test_score': test_score,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'n_features': X_train.shape[1],
                'training_date': datetime.now().isoformat()
            }
            
            return metrics, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    
    def save_model(self, metrics):
        """Save the trained model and encoders"""
        try:
            logger.info("Saving model and encoders...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_filename = f"xgboost_model_{timestamp}.pkl"
            model_filepath = self.model_path / model_filename
            
            with open(model_filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to: {model_filepath}")
            
            # Save label encoders
            encoders_filename = f"label_encoders_{timestamp}.pkl"
            encoders_filepath = self.model_path / encoders_filename
            
            with open(encoders_filepath, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            logger.info(f"Label encoders saved to: {encoders_filepath}")
            
            # Save metrics
            metrics_filename = f"model_metrics_{timestamp}.pkl"
            metrics_filepath = self.model_path / metrics_filename
            
            with open(metrics_filepath, 'wb') as f:
                pickle.dump(metrics, f)
            logger.info(f"Metrics saved to: {metrics_filepath}")
            
            # Save latest model info
            latest_info = {
                'model_path': str(model_filepath),
                'encoders_path': str(encoders_filepath),
                'metrics_path': str(metrics_filepath),
                'timestamp': timestamp,
                'metrics': metrics
            }
            
            latest_filepath = self.model_path / "latest_model_info.pkl"
            with open(latest_filepath, 'wb') as f:
                pickle.dump(latest_info, f)
            
            return latest_info
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info("🚀 Starting ML training pipeline...")
            
            # Load data
            df = self.load_data()
            
            # Preprocess data
            df_processed, target_col = self.preprocess_data(df)
            
            # Train model
            metrics, X_test, y_test = self.train_model(df_processed, target_col)
            
            # Save model
            model_info = self.save_model(metrics)
            
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"Model info: {model_info}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

def main():
    """Main function to run the training"""
    try:
        # Initialize trainer with 1 million records limit
        trainer = BeverageMLTrainer(max_records=1000000)
        
        # Run training pipeline
        model_info = trainer.run_training_pipeline()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model saved at: {model_info['model_path']}")
        print(f"Training R² Score: {model_info['metrics']['train_score']:.4f}")
        print(f"Test R² Score: {model_info['metrics']['test_score']:.4f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
