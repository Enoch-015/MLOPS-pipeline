# MLOPS-pipeline

A complete MLOps pipeline setup with ClearML for experiment tracking, model management, and workflow orchestration.

## 🚀 Quick Start

### ClearML Server
The repository includes a full ClearML server setup using Docker Compose:

```bash
# Start all ClearML services
docker-compose up -d

# Verify installation
python test_clearml.py
```

**Access Points:**
- 🌐 **Web Interface**: http://localhost:8080
- 🔌 **API Server**: http://localhost:8008  
- 📁 **File Server**: http://localhost:8081

### ClearML Client Setup
```bash
# Install ClearML Python package
pip install -r requirements.txt

# Configure (get credentials from web interface)
clearml-init
```

## 💻 ClearML Pipeline Implementation

### Testing the Pipeline

You can test individual steps of the pipeline using the `test_pipeline.py` script:

```bash
# Test extract step
python -m src.test_pipeline extract

# Test preprocessing step
python -m src.test_pipeline preprocessing --dataset-id <dataset_id>

# Test training step
python -m src.test_pipeline train --dataset-id <dataset_id>

# Test evaluation step
python -m src.test_pipeline evaluate --model-id <model_id> --dataset-id <dataset_id>

# Test deployment step
python -m src.test_pipeline deploy --model-id <model_id>

# Test all steps in sequence
python -m src.test_pipeline all
```

### Running the Complete Pipeline

To run the complete pipeline, use:

```bash
python -m src.pipelines.pipeline
```

### Pipeline Steps

1. **Extract**: Connects to the database and extracts beverage sales data
   - Handles database connection failures with synthetic data generation
   - Creates a versioned ClearML dataset with the raw data

2. **Preprocess**: Cleans data and prepares for modeling
   - Handles datetime features
   - Encodes categorical variables
   - Removes missing values
   - Creates a versioned ClearML dataset with processed data

3. **Train**: Trains an XGBoost model
   - Uses grid search for hyperparameter optimization
   - Tracks model metrics during training
   - Saves the best model as a ClearML artifact

4. **Evaluate**: Evaluates model performance
   - Generates performance metrics (RMSE, MAE, R²)
   - Creates visualization artifacts (prediction plots)
   - Logs evaluation results to ClearML

5. **Deploy**: Packages model for deployment
   - Creates a deployment-ready model package
   - Logs model serving requirements

### Project Structure

```
├── src/
│   ├── ETL/              # Database connection and ETL utilities
│   ├── steps/            # Individual pipeline steps
│   │   ├── extract.py       # Data extraction from database
│   │   ├── preprocessing.py # Data cleaning and preparation
│   │   ├── train.py         # Model training
│   │   ├── evaluate.py      # Model evaluation
│   │   ├── deploy.py        # Model deployment
│   │   └── config.py        # Configuration for all steps
│   ├── pipelines/        # Pipeline orchestration
│   │   └── pipeline.py      # Main pipeline definition
│   └── ml_training.py    # Original ML training code (for reference)
├── docker-compose.yml    # Docker Compose config for ClearML server
├── requirements.txt      # Python dependencies
└── test_pipeline.py      # Script for testing individual pipeline steps
```

## 📚 Documentation
- See [CLEARML_SETUP.md](CLEARML_SETUP.md) for detailed setup instructions
- See [ClearML Documentation](https://clear.ml/docs) for usage guides

## 🛠 Services Included
- **Web UI**: Experiment dashboard and model registry
- **API Server**: RESTful API for programmatic access
- **File Server**: Artifact and dataset storage
- **Agent Services**: Distributed training capabilities
- **Databases**: MongoDB, Redis, Elasticsearch

## 🎯 Features
- ✅ Experiment tracking and comparison
- ✅ Model versioning and registry
- ✅ Dataset management
- ✅ Hyperparameter optimization
- ✅ Distributed training orchestration
- ✅ CI/CD pipeline integration

## 📈 Model Performance Tracking

ClearML automatically tracks model performance metrics throughout the pipeline:

1. **Training Metrics**:
   - Loss curves
   - Feature importance
   - Hyperparameter influence

2. **Evaluation Metrics**:
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - R² (Coefficient of Determination)
   - Prediction vs. Actual plots

3. **Resource Utilization**:
   - CPU/GPU usage
   - Memory consumption
   - Training time

All metrics are available through the ClearML web interface for easy comparison between experiments.

## 🔍 Troubleshooting

If you encounter issues:

1. **ClearML Server Connection Problems**:
   ```bash
   # Check server status
   docker-compose ps
   
   # View server logs
   docker-compose logs -f
   ```

2. **Database Connection Issues**:
   - Verify credentials in `.env` file
   - Ensure database is accessible from your environment
   - Check firewall settings if accessing remote database

3. **Pipeline Errors**:
   - Use the test_pipeline.py script to debug individual steps
   - Check ClearML web UI for detailed error messages and logs