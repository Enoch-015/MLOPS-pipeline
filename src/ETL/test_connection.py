"""
Database connection test utility
"""

from sqlalchemy import create_engine, text
import logging
from config import DB_CONNECTION_STRING

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test database connection"""
    try:
        logger.info("Testing database connection...")
        
        # Create engine
        engine = create_engine(DB_CONNECTION_STRING)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ Connected successfully!")
            logger.info(f"PostgreSQL version: {version}")
            
            # Test basic operations
            result = conn.execute(text("SELECT current_database(), current_user"))
            db_info = result.fetchone()
            logger.info(f"Database: {db_info[0]}")
            logger.info(f"User: {db_info[1]}")
            
            # List existing tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = result.fetchall()
            
            if tables:
                logger.info("Existing tables:")
                for table in tables:
                    logger.info(f"  - {table[0]}")
            else:
                logger.info("No tables found in public schema")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if not success:
        exit(1)
