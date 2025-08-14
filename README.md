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
pip install clearml

# Configure (get credentials from web interface)
clearml-init
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