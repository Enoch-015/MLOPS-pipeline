"""
ClearML Pipeline using function decorators, built on pure step functions.

This module imports step implementations from src/steps and exposes them
as ClearML pipeline components using PipelineDecorator. The main pipeline
function orchestrates the flow.
"""

import logging
from pathlib import Path
from datetime import datetime

from clearml.automation.controller import PipelineDecorator

# Optional: basic logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Components
@PipelineDecorator.component(return_values=["raw_df"], cache=True, repo="./")
def step_extract(max_records: int = 50000):
    """Extract data to a pandas DataFrame."""
    import logging
    _logger = logging.getLogger("pipeline.step_extract")
    # Ensure we can import our local src modules
    import sys
    from pathlib import Path
    for p in [Path.cwd() / "src", Path.cwd()]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # Import inside the component to work in subprocess context
    try:
        from src.steps.extract import extract_data  # type: ignore
    except Exception:
        from steps.extract import extract_data  # type: ignore

    import pandas as pd  # noqa: F401 (ensures package capture)
    df = extract_data(max_records=max_records)
    try:
        _logger.info(f"step_extract returned df shape: {getattr(df, 'shape', None)}")
    except Exception:
        pass
    return df


@PipelineDecorator.component(return_values=["df_processed", "metadata"], cache=False, repo="./")
def step_preprocess(raw_df):
    """Preprocess raw dataframe and return processed df and metadata."""
    import logging
    _logger = logging.getLogger("pipeline.step_preprocess")
    # Import inside the component to work in subprocess context
    import sys
    from pathlib import Path
    for p in [Path.cwd() / "src", Path.cwd()]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from src.steps.preprocessing import preprocess_data  # type: ignore
    except Exception:
        from steps.preprocessing import preprocess_data  # type: ignore

    import pandas as pd  # noqa
    df_processed, metadata = preprocess_data(raw_df)
    try:
        _logger.info(f"step_preprocess returning metadata: {metadata}")
    except Exception:
        pass
    return df_processed, metadata


@PipelineDecorator.component(return_values=["model", "train_metrics", "feature_importance"], cache=False, repo="./")
def step_train(df_processed, metadata: dict):
    """Train model and return model object, training metrics, and feature importance."""
    import logging
    _logger = logging.getLogger("pipeline.step_train")
    # Import inside the component to work in subprocess context
    import sys
    from pathlib import Path
    for p in [Path.cwd() / "src", Path.cwd()]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from src.steps.train import train_model  # type: ignore
    except Exception:
        from steps.train import train_model  # type: ignore
    from xgboost import XGBRegressor  # noqa
    
    # Extract target column from metadata
    target_col = metadata['target_column']
    try:
        _logger.info(f"step_train extracted target_col: '{target_col}' from metadata")
    except Exception:
        pass
    
    model, metrics, feature_importance = train_model(df_processed, target_col)
    return model, metrics, feature_importance


@PipelineDecorator.component(return_values=["evaluation_results"], cache=False, repo="./")
def step_evaluate(model, df_processed, metadata: dict):
    """Evaluate model and return evaluation results dict."""
    import logging
    _logger = logging.getLogger("pipeline.step_evaluate")
    # Import inside the component to work in subprocess context
    import sys
    from pathlib import Path
    for p in [Path.cwd() / "src", Path.cwd()]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from src.steps.evaluate import evaluate_model  # type: ignore
    except Exception:
        from steps.evaluate import evaluate_model  # type: ignore
    from sklearn.metrics import mean_squared_error  # noqa
    
    # Extract target column from metadata
    target_col = metadata['target_column']
    try:
        _logger.info(f"step_evaluate extracted target_col: '{target_col}' from metadata")
    except Exception:
        pass
    
    results = evaluate_model(model, df_processed, target_col)
    return results


@PipelineDecorator.component(return_values=["deployment_dir"], cache=False, repo="./")
def step_deploy(model, metadata: dict, evaluation_results):
    """Package model for deployment and return local deployment directory path."""
    import logging
    _logger = logging.getLogger("pipeline.step_deploy")
    # Import inside the component to work in subprocess context
    import sys
    from pathlib import Path
    for p in [Path.cwd() / "src", Path.cwd()]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from src.steps.deploy import deploy_model  # type: ignore
    except Exception:
        from steps.deploy import deploy_model  # type: ignore
    # Extract necessary information from metadata
    target_col = metadata['target_column']
    label_encoders = metadata['label_encoders']
    
    try:
        _logger.info(f"step_deploy extracted target_col: '{target_col}' from metadata")
    except Exception:
        pass
    
    path = deploy_model(model, label_encoders, evaluation_results, target_col)
    return path


# Pipeline Controller
@PipelineDecorator.pipeline(
    name="beverage_sales_pipeline",
    project="MLOps-Beverage-Sales",
    version="0.1.0",
    args_map={"General": ["max_records"]},
)
def main(max_records: int = 50000):
    logger.info("Starting pipeline execution...")
    
    # 1) Extract
    raw_df = step_extract(max_records=max_records)

    # 2) Preprocess - now returns structured metadata
    df_processed, metadata = step_preprocess(raw_df)
    
    # Log metadata for debugging
    logger.info(f"Pipeline main received metadata: {metadata}")
    logger.info(f"Target column: '{metadata['target_column']}'")

    # 3) Train - pass metadata instead of individual target_col
    model, train_metrics, feature_importance = step_train(df_processed, metadata)

    # 4) Evaluate - pass metadata
    evaluation_results = step_evaluate(model, df_processed, metadata)

    # 5) Deploy - pass metadata (contains both label_encoders and target_col)
    deployment_dir = step_deploy(model, metadata, evaluation_results)

    logger.info("Pipeline execution completed successfully")
    print("Pipeline completed")
    print(f"Train metrics: {train_metrics}")
    print(f"Evaluation metrics: {evaluation_results.get('metrics')}")
    print(f"Deployment dir: {deployment_dir}")


if __name__ == "__main__":
    # For quick local debugging:
    # Run the pipeline controller locally while components can still run remote if configured
    # PipelineDecorator.debug_pipeline()  # Uncomment for fully synchronous local debugging
    main(max_records=5000)