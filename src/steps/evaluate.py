"""
Model Evaluation Step

Pure function to evaluate a trained model on provided data.
Returns metrics and paths to generated plots (if any). No ClearML Task code.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import configuration
from config import LOG_CONFIG, LOGS_DIR, ARTIFACTS_DIR

# Setup logging
log_file = LOGS_DIR / f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    if isinstance(value, str) and value in columns:
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _normalize_target_col(value[0], columns)
    if isinstance(value, (list, tuple)) and value and all(isinstance(c, str) and len(c) == 1 for c in value):
        joined = "".join(value)
        if joined in columns:
            return joined
        for col in columns:
            if col.lower() == joined.lower():
                return col
        return joined
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return _normalize_target_col(value[0], columns)
    try:
        import numpy as _np
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

def evaluate_model(model, df_processed: pd.DataFrame, target_col: str):
    """Evaluate the trained model.

    Args:
        model: Trained model with predict method
        df_processed: Preprocessed dataframe including target column
        target_col: Name of the target column

    Returns:
        dict: Evaluation results containing metrics and generated artifacts paths
    """
    target_col = _normalize_target_col(target_col, df_processed.columns)
    logging.getLogger(__name__).info(f"Resolved target column: {target_col}")

    # Separate features and target
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X)

    # Calculate metrics
    metrics = {
        'mse': float(mean_squared_error(y, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2': float(r2_score(y, y_pred)),
        'explained_variance': float(explained_variance_score(y, y_pred))
    }

    logger.info("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

    # Create a scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')

    # Create temp directory for plots
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    # Save the plot
    scatter_plot_path = temp_dir / "actual_vs_predicted.png"
    plt.savefig(scatter_plot_path)
    plt.close()

    # Create histogram of residuals
    residuals = y - y_pred
    plt.figure(figsize=(10, 8))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')

    # Save the residuals plot
    residuals_plot_path = temp_dir / "residuals_distribution.png"
    plt.savefig(residuals_plot_path)
    plt.close()

    eval_results = {
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'artifacts': {
            'actual_vs_predicted': str(scatter_plot_path),
            'residuals_distribution': str(residuals_plot_path)
        }
    }

    logger.info("Model evaluation completed successfully")

    return eval_results

if __name__ == "__main__":
    print("This module provides evaluate_model(model, df_processed, target_col) pure function.")
