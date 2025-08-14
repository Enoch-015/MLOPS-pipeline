# ClearML MLOps Pipeline Setup

## Overview
This repository contains a complete ClearML setup for MLOps pipeline management using Docker.

## Services Running
- **Web Interface**: http://localhost:8080 - Main ClearML web dashboard
- **API Server**: http://localhost:8008 - REST API for programmatic access
- **File Server**: http://localhost:8081 - Artifact and model storage
- **Database Services**: MongoDB, Redis, Elasticsearch (internal)
- **Agent Services**: For running experiments

## Quick Start

### 1. Start ClearML Services
```bash
docker-compose up -d
```

### 2. Stop ClearML Services
```bash
docker-compose down
```

### 3. Check Service Status
```bash
docker-compose ps
```

### 4. View Logs
```bash
# All services
docker-compose logs

# Specific service
docker logs clearml-webserver
docker logs clearml-apiserver
docker logs clearml-elastic
```

## Installation Verification

Run the test script to verify all services are working:
```bash
python test_clearml.py
```

## ClearML Client Setup

### Install ClearML Python Package
```bash
pip install clearml
```

### Configure ClearML Client
1. Access the web interface at http://localhost:8080
2. Go to Settings → Workspace → App Credentials
3. Create new credentials
4. Run: `clearml-init` and paste the configuration

### Example Usage
```python
from clearml import Task

# Initialize a task
task = Task.init(project_name="MyProject", task_name="MyExperiment")

# Your ML code here
# ClearML will automatically track:
# - Git repository and commit
# - Python packages
# - Hyperparameters
# - Metrics and scalars
# - Models and artifacts

print("Hello ClearML!")
```

## Directory Structure
```
/opt/clearml/
├── logs/           # Service logs
├── config/         # Configuration files
├── data/
│   ├── fileserver/ # File storage
│   ├── elastic_7/  # Elasticsearch data
│   ├── mongo_4/    # MongoDB data
│   └── redis/      # Redis data
└── agent/          # Agent configuration
```

## Environment Variables
Edit `.env` file to customize:
- `CLEARML_HOST_IP`: Host IP address
- `CLEARML_WEB_HOST`: Web interface URL
- `CLEARML_FILES_HOST`: File server URL
- API keys and credentials (optional)

## Common Commands

### Docker Management
```bash
# View running containers
docker ps

# Restart a specific service
docker-compose restart clearml-webserver

# Update to latest images
docker-compose pull
docker-compose up -d

# Clean up (removes containers and data)
docker-compose down -v
```

### Troubleshooting
```bash
# Check if Elasticsearch is healthy
curl http://localhost:9200/_cluster/health

# Test API connectivity
curl http://localhost:8008/debug.ping

# Check disk usage
df -h /opt/clearml/
```

## System Requirements
- Docker and Docker Compose
- At least 4GB RAM (8GB+ recommended)
- 10GB+ free disk space
- `vm.max_map_count=262144` (already configured)

## Security Notes
- Default setup is for development/local use
- For production, configure authentication and SSL
- Restrict network access appropriately
- Regular backups of `/opt/clearml/data/`

## Next Steps
1. Install ClearML Python client: `pip install clearml`
2. Access web interface: http://localhost:8080
3. Create your first project and experiment
4. Set up agents for distributed training (optional)

## Support
- ClearML Documentation: https://clear.ml/docs
- GitHub Issues: https://github.com/allegroai/clearml/issues
- Community Slack: https://joinslack.clear.ml/
