"""
Model Deployment Step

This step prepares the model for deployment by packaging it with its artifacts.
"""
"""
Model Deployment Step

Pure function to package the trained model and supporting files for deployment.
Returns the local path to the deployment package directory.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import os
import json
import shutil
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import configuration
from config import LOG_CONFIG, LOGS_DIR, MODELS_DIR, ARTIFACTS_DIR

# Setup logging
log_file = LOGS_DIR / f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['level']),
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def deploy_model(model, label_encoders: Optional[dict], evaluation_results: dict, target_col: str = "Total_Price"):
    """Prepare the model for deployment by packaging it and its metadata.

    Args:
        model: Trained model object
        label_encoders: Optional dict of fitted encoders
        evaluation_results: Dict from evaluate_model
        target_col: Target column name

    Returns:
        str: Local path to the deployment package directory
    """
    # Derive a model name and path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_model_{timestamp}"

    # Create deployment package directory
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    deploy_dir = temp_dir / f"deployment_{timestamp}"
    os.makedirs(deploy_dir, exist_ok=True)

    # Save model file to deployment directory
    model_file = deploy_dir / f"{model_name}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to: {model_file}")

    # Save label encoders if provided
    if label_encoders:
        encoders_path = deploy_dir / "label_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        logger.info("Saved label encoders to deployment package")

    # Create simplified model card with available information
    model_card = {
        "model_name": model_name,
        "model_type": "XGBoost Regressor",
        "created_at": datetime.now().isoformat(),
        "target_column": target_col,
        "metrics": evaluation_results.get('metrics', {}),
        "deployment_timestamp": timestamp
    }

    # Save model card as JSON
    model_card_path = deploy_dir / "model_card.json"
    with open(model_card_path, 'w') as f:
        json.dump(model_card, f, indent=4)

    logger.info(f"Model card created: {model_card_path}")

    # Create a simple prediction script
    prediction_script = """
import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_model(model_path=None):
    \"\"\"Load the trained model and encoders\"\"\"
    if model_path is None:
        model_path = Path(__file__).parent
    
    # Find model file
    model_files = list(Path(model_path).glob("*.pkl"))
    model_file = [f for f in model_files if "label_encoders" not in f.name][0]
    
    # Load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load encoders if available
    encoders_path = Path(model_path) / "label_encoders.pkl"
    encoders = None
    if encoders_path.exists():
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
    
    # Load model card
    model_card_path = Path(model_path) / "model_card.json"
    model_card = None
    if model_card_path.exists():
        import json
        with open(model_card_path, 'r') as f:
            model_card = json.load(f)
    
    return model, encoders, model_card

def preprocess_input(input_data, encoders, model_card):
    \"\"\"Preprocess input data for prediction\"\"\"
    # Convert input to DataFrame if it's a dict
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Check if we have encoders for categorical columns
    if encoders and model_card:
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))
    
    return input_data

def predict(input_data, model_path=None):
    \"\"\"Make predictions using the trained model\"\"\"
    # Load model and encoders
    model, encoders, model_card = load_model(model_path)
    
    # Preprocess the input
    processed_input = preprocess_input(input_data, encoders, model_card)
    
    # Make prediction
    prediction = model.predict(processed_input)
    
    return prediction

if __name__ == "__main__":
    # Example usage
    sample_input = {
        # Add sample input features here
    }
    result = predict(sample_input)
    print(f"Prediction: {result}")
"""
    # Save prediction script
    predict_script_path = deploy_dir / "predict.py"
    with open(predict_script_path, 'w') as f:
        f.write(prediction_script)

    # Create a README
    readme_content = f"""# {model_name} Deployment Package

This package contains the trained model and all necessary files for deployment.

## Files:
- `{model_name}.pkl`: The trained XGBoost model
- `label_encoders.pkl`: Label encoders for categorical features
- `model_card.json`: Model metadata and performance metrics
- `predict.py`: Python script for making predictions

## Usage:
```python
from predict import predict

# Example input (adjust features based on your model)
sample_input = {{
    'feature1': value1,
    'feature2': value2,
    # ...
}}

# Get prediction
result = predict(sample_input)
print(f"Prediction: {{result}}")
```

## Model Performance:
- R² Score: {evaluation_results.get('metrics', {}).get('r2', 'N/A')}
- RMSE: {evaluation_results.get('metrics', {}).get('rmse', 'N/A')}
- MAE: {evaluation_results.get('metrics', {}).get('mae', 'N/A')}

## Deployment Date:
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    # Save README
    readme_path = deploy_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    logger.info(f"Created deployment README: {readme_path}")

    # Also copy to the local artifacts directory for easy access
    local_deploy_dir = ARTIFACTS_DIR / f"deployment_{timestamp}"
    if not local_deploy_dir.exists():
        shutil.copytree(deploy_dir, local_deploy_dir)

    logger.info("Model deployed locally successfully")
    logger.info(f"Deployment package: {local_deploy_dir}")

    return str(local_deploy_dir)

if __name__ == "__main__":
    print("This module provides deploy_model(model, label_encoders, evaluation_results, target_col) pure function.")
