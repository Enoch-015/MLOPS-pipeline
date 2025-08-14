# Environment Configuration

This document explains how to set up environment variables for secure configuration management.

## Setup Instructions

### 1. Environment Variables File

Copy the example environment file and customize it with your values:

```bash
cp .env.example .env
```

Then edit `.env` with your actual configuration values.

### 2. Required Environment Variables

#### Database Configuration
- `DB_HOST`: Database host (e.g., `localhost` or `your-db-server.com`)
- `DB_PORT`: Database port (default: `5432`)
- `DB_NAME`: Database name
- `DB_USERNAME`: Database username
- `DB_PASSWORD`: Database password

#### ETL Configuration
- `ETL_CHUNK_SIZE`: Size of data chunks for processing (default: `10000`)
- `ETL_TABLE_NAME`: Target table name (default: `beverage_sales`)
- `ETL_BATCH_SIZE`: Batch size for database operations (default: `1000`)
- `ETL_MAX_RETRIES`: Maximum retry attempts (default: `3`)
- `ETL_RETRY_DELAY`: Delay between retries in seconds (default: `5`)

#### Logging Configuration
- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### 3. Production Deployment

For production environments:

1. **Never commit `.env` files** - they are gitignored for security
2. **Use environment-specific files** like `.env.prod`, `.env.staging`
3. **Set environment variables directly** in your deployment platform
4. **Use secrets management** services for sensitive data

### 4. Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment file
cp .env.example .env
# Edit .env with your actual values

# Test the configuration
python src/ETL/test_connection.py
```

## Security Notes

- The `.env` file contains sensitive information and is automatically ignored by git
- Use different credentials for development, staging, and production
- Regularly rotate database passwords and API keys
- Never hardcode sensitive values in your source code

## Files Ignored by Git

- `src/data/raw/` - Raw data files
- `src/data/interim/` - Intermediate processing files  
- `src/data/processed/*.csv` - Processed CSV files
- `src/data/processed/*.json` - Processed JSON files
- `src/data/processed/*.parquet` - Processed Parquet files
- `.env` - Environment variables file
- `.env.local`, `.env.prod`, `.env.dev` - Environment-specific files
