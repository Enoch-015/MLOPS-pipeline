# MLOps Pipeline with ClearML

This project implements an end-to-end MLOps pipeline using ClearML for training, evaluating, and deploying a machine learning model for beverage sales prediction.

## Project Structure

```
MLOPS-pipeline/
│
├── run_pipeline.py        # Main script to run the complete pipeline
├── src/
│   ├── ETL/               # ETL scripts and configuration
│   ├── steps/             # ClearML pipeline steps
│   │   ├── config.py      # Common configuration for all steps
│   │   ├── extract.py     # Data extraction step
│   │   ├── preprocessing.py # Data preprocessing step
│   │   ├── train.py       # Model training step
│   │   ├── evaluate.py    # Model evaluation step
│   │   └── deploy.py      # Model deployment step
│   ├── pipelines/         # Pipeline definitions
│   │   └── pipeline.py    # Main pipeline definition
│   ├── models/            # Directory for saved models
│   └── artifacts/         # Directory for model artifacts and deployment packages
├── logs/                  # Log files from pipeline steps
└── clearml-server/        # ClearML server files
```

## Setup

1. Make sure the ClearML server is running:

```bash
cd clearml-server/docker && docker-compose up -d
```

2. Configure ClearML client:

```bash
clearml-init
```

3. Set up environment variables (or use the .env file):

```bash
export DB_HOST=your-db-host
export DB_PORT=your-db-port
export DB_NAME=your-db-name
export DB_USERNAME=your-username
export DB_PASSWORD=your-password
```

## Running the Pipeline

To run the complete pipeline locally:

```bash
python run_pipeline.py --max-records 50000
```

To run with more options:

```bash
python run_pipeline.py --max-records 10000 --log-level DEBUG
```

To schedule the pipeline on a remote ClearML agent:

```bash
python run_pipeline.py --remote --max-records 1000000
```

## Pipeline Steps

1. **Data Extraction**: Extracts data from PostgreSQL database
2. **Data Preprocessing**: Cleans and prepares data for modeling
3. **Model Training**: Trains an XGBoost regression model
4. **Model Evaluation**: Evaluates model performance on test data
5. **Model Deployment**: Prepares the model for deployment

## Data Versioning

The pipeline uses ClearML Datasets for data versioning. Each dataset is tracked with a unique ID and maintains lineage information.

## Model Versioning

Models are versioned and tracked through ClearML Models. Each model version includes:
- Model weights
- Performance metrics
- Training parameters
- Dataset lineage

## Artifacts

All artifacts generated during the pipeline (datasets, models, evaluation results) are stored and tracked through ClearML.

## Logging

Detailed logs are written to the `logs/` directory and are also available in the ClearML UI.

## License

[Include your license information here]
