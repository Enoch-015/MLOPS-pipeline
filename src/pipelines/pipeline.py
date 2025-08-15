"""
End-to-end MLOps Pipeline with ClearML

This script ties together all the pipeline steps for training and deploying a machine learning model.
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime
from clearml import Task, PipelineController

# Import the steps
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.steps.config import LOG_CONFIG, CLEARML_PROJECT_NAME, LOGS_DIR
from src.steps.extract import extract_data
from src.steps.preprocessing import preprocess_data
from src.steps.train import train_model
from src.steps.evaluate import evaluate_model
from src.steps.deploy import deploy_model

# Setup logging
log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_and_run_pipeline(max_records=50000, run_locally=True):
    """
    Create and run the complete MLOps pipeline
    
    Args:
        max_records (int): Maximum number of records to use for training
        run_locally (bool): If True, run the pipeline locally, otherwise schedule on a ClearML agent
    """
    # Create pipeline controller
    pipe = PipelineController(
        name="Beverage Sales MLOps Pipeline",
        project=CLEARML_PROJECT_NAME,
        version="1.0",
        add_pipeline_tags=True
    )
    
    # Add the steps to the pipeline
    pipe.add_step(
        name="extract_data",
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="Data Extraction",
        parameter_override={"General/max_records": max_records},
        execution_queue="default" if not run_locally else None
    )
    
    pipe.add_step(
        name="preprocess_data",
        parents=["extract_data"],
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="Data Preprocessing",
        parameter_override={"General/dataset_id": "${extract_data.artifacts.dataset_id}"},
        execution_queue="default" if not run_locally else None
    )
    
    pipe.add_step(
        name="train_model",
        parents=["preprocess_data"],
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="Model Training",
        parameter_override={
            "General/preprocessed_dataset_id": "${preprocess_data.artifacts.preprocessed_dataset_id}",
            "General/target_col": "${preprocess_data.artifacts.target_column}"
        },
        execution_queue="default" if not run_locally else None
    )
    
    pipe.add_step(
        name="evaluate_model",
        parents=["train_model"],
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="Model Evaluation",
        parameter_override={
            "General/model_id": "${train_model.artifacts.model_id}",
            "General/preprocessed_dataset_id": "${preprocess_data.artifacts.preprocessed_dataset_id}",
            "General/target_col": "${preprocess_data.artifacts.target_column}"
        },
        execution_queue="default" if not run_locally else None
    )
    
    pipe.add_step(
        name="deploy_model",
        parents=["evaluate_model"],
        base_task_project=CLEARML_PROJECT_NAME,
        base_task_name="Model Deployment",
        parameter_override={
            "General/model_id": "${train_model.artifacts.model_id}",
            "General/preprocessed_dataset_id": "${preprocess_data.artifacts.preprocessed_dataset_id}",
            "General/evaluation_results": "${evaluate_model.artifacts.evaluation_results}"
        },
        execution_queue="default" if not run_locally else None
    )
    
    # Start the pipeline
    pipe.start_locally(run_pipeline_steps_locally=run_locally)
    
    try:
        pipeline_id = pipe.pipeline_id if hasattr(pipe, 'pipeline_id') else pipe.id
        logger.info(f"Pipeline started with ID: {pipeline_id}")
        return pipe
    except AttributeError:
        logger.info("Pipeline started successfully")
        return pipe

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="Run MLOps Pipeline")
    parser.add_argument(
        "--max-records", 
        type=int, 
        default=50000,
        help="Maximum number of records to load for training"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run pipeline on remote ClearML agent instead of locally"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting MLOps Pipeline...")
        logger.info(f"Max records: {args.max_records}")
        logger.info(f"Running {'remotely' if args.remote else 'locally'}")
        
        pipe = create_and_run_pipeline(
            max_records=args.max_records,
            run_locally=not args.remote
        )
        
        logger.info("🎉 Pipeline created and started!")
        try:
            pipeline_id = pipe.pipeline_id if hasattr(pipe, 'pipeline_id') else pipe.id
            logger.info(f"Pipeline ID: {pipeline_id}")
        except AttributeError:
            logger.info("Pipeline started successfully")
        
    except Exception as e:
        logger.error(f"Pipeline creation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
