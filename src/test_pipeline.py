#!/usr/bin/env python
"""
Test Pipeline Script

This script provides a simplified way to test the individual steps of our ClearML pipeline.
It allows testing of individual components without running the full pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def test_extract():
    """Test the extract step in isolation"""
    from src.steps.extract import extract_data
    
    logger.info("Testing extract step...")
    dataset = extract_data(max_records=100)
    
    # Log some information about the dataset
    if dataset:
        logger.info(f"Dataset ID: {dataset.id}")
        logger.info(f"Dataset name: {dataset.name}")
        
        # Try to get the local copy
        local_copy = dataset.get_local_copy()
        logger.info(f"Local dataset path: {local_copy}")
    else:
        logger.error("Dataset creation failed")
    
    return dataset

def test_preprocessing(dataset_id=None):
    """Test the preprocessing step in isolation"""
    from src.steps.preprocessing import preprocess_data
    from clearml import Dataset
    
    if not dataset_id:
        # Create a dataset from extract step
        logger.info("No dataset ID provided, running extract step first...")
        dataset = test_extract()
        dataset_id = dataset.id if dataset else None
        
    if dataset_id:
        logger.info(f"Testing preprocessing step with dataset ID: {dataset_id}...")
        try:
            processed_dataset = preprocess_data(dataset_id)
            
            # Log some information about the processed dataset
            if processed_dataset:
                logger.info(f"Processed dataset ID: {processed_dataset.id}")
                logger.info(f"Processed dataset name: {processed_dataset.name}")
                
                # Try to get the local copy
                local_copy = processed_dataset.get_local_copy()
                logger.info(f"Local processed dataset path: {local_copy}")
            else:
                logger.error("Processed dataset creation failed")
            
            return processed_dataset
        except ValueError as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return None
    else:
        logger.error("No dataset ID available for preprocessing")
        return None

def test_train(dataset_id=None):
    """Test the training step in isolation"""
    from src.steps.train import train_model
    from clearml import Dataset
    
    if not dataset_id:
        # Create a dataset from preprocessing step
        logger.info("No dataset ID provided, running preprocessing step first...")
        processed_dataset = test_preprocessing()
        dataset_id = processed_dataset.id if processed_dataset else None
        
    if dataset_id:
        logger.info(f"Testing training step with dataset ID: {dataset_id}...")
        # Use the default target column from our beverage sales data
        target_col = "Total_Price"
        trained_model = train_model(dataset_id, target_col)
        
        # Log some information about the trained model
        if trained_model:
            if hasattr(trained_model, 'id'):
                logger.info(f"Trained model ID: {trained_model.id}")
            if hasattr(trained_model, 'name'):
                logger.info(f"Trained model name: {trained_model.name}")
            else:
                logger.info(f"Training completed successfully")
        else:
            logger.error("Model training failed")
        
        return trained_model
    else:
        logger.error("No dataset ID available for training")
        return None

def test_evaluate(model_id=None, dataset_id=None):
    """Test the evaluation step in isolation"""
    from src.steps.evaluate import evaluate_model
    
    if not model_id or not dataset_id:
        # Create a model from training step
        logger.info("No model or dataset ID provided, running training step first...")
        trained_model = test_train()
        if trained_model and hasattr(trained_model, 'id'):
            # Get model ID from training task artifacts
            model_id = "http://localhost:8081/MLOps-Beverage-Sales/Model Training.{}/models/".format(trained_model.id)
            # For dataset ID, we'll use the preprocessed dataset from our previous test
            dataset_id = "b5d43606ab07431aabdaddb46142beab"  # Use the known preprocessed dataset
        
    if model_id and dataset_id:
        logger.info(f"Testing evaluation step with model ID: {model_id} and dataset ID: {dataset_id}...")
        target_col = "Total_Price"
        try:
            evaluation_task = evaluate_model(model_id, dataset_id, target_col)
            
            # Log some information about the evaluation task
            if evaluation_task:
                logger.info(f"Evaluation task ID: {evaluation_task.id}")
                logger.info(f"Evaluation task name: {evaluation_task.name}")
            else:
                logger.error("Model evaluation failed")
            
            return evaluation_task
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            return None
    else:
        logger.error("No model ID or dataset ID available for evaluation")
        return None

def test_deploy(model_id=None):
    """Test the deployment step in isolation"""
    from src.steps.deploy import deploy_model
    
    if not model_id:
        # Create a model from training step
        logger.info("No model ID provided, running training step first...")
        trained_model = test_train()
        if trained_model and hasattr(trained_model, 'id'):
            # Get model ID from training task
            model_id = "http://localhost:8081/MLOps-Beverage-Sales/Model Training.{}/models/".format(trained_model.id)
        
    if model_id:
        logger.info(f"Testing deployment step with model ID: {model_id}...")
        
        # Use known preprocessed dataset ID and create mock evaluation results
        preprocessed_dataset_id = "b5d43606ab07431aabdaddb46142beab"
        evaluation_results = {
            'metrics': {
                'r2': 0.8533,
                'rmse': 268.94,
                'mae': 46.00,
                'mse': 72327.79,
                'explained_variance': 0.8565
            }
        }
        
        try:
            deployment_task = deploy_model(model_id, preprocessed_dataset_id, evaluation_results)
            
            # Log some information about the deployment task
            if deployment_task:
                logger.info(f"Deployment task ID: {deployment_task.id}")
                logger.info(f"Deployment task name: {deployment_task.name}")
            else:
                logger.error("Model deployment failed")
            
            return deployment_task
        except Exception as e:
            logger.error(f"Error in deployment: {str(e)}")
            return None
    else:
        logger.error("No model ID available for deployment")
        return None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test individual pipeline steps")
    parser.add_argument('step', type=str, choices=['extract', 'preprocessing', 'train', 'evaluate', 'deploy', 'all'],
                      help='The pipeline step to test')
    parser.add_argument('--dataset-id', type=str, help='Dataset ID to use for testing', default=None)
    parser.add_argument('--model-id', type=str, help='Model ID to use for testing', default=None)
    return parser.parse_args()

def main():
    """Main function to test pipeline steps"""
    args = parse_args()
    
    logger.info(f"Testing step: {args.step}")
    
    if args.step == 'extract' or args.step == 'all':
        test_extract()
        
    if args.step == 'preprocessing' or args.step == 'all':
        test_preprocessing(args.dataset_id)
        
    if args.step == 'train' or args.step == 'all':
        test_train(args.dataset_id)
        
    if args.step == 'evaluate' or args.step == 'all':
        test_evaluate(args.model_id, args.dataset_id)
        
    if args.step == 'deploy' or args.step == 'all':
        test_deploy(args.model_id)
        
    logger.info("Testing completed")

if __name__ == "__main__":
    main()
