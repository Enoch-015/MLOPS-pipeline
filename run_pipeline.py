#!/usr/bin/env python
"""
MLOps Pipeline Runner (Function-Decorators)

Entry point to run the ClearML pipeline defined with PipelineDecorator in src/pipeline.py
Supports remote, local, and debug modes.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))

from clearml.automation.controller import PipelineDecorator  # noqa: E402
import pipeline as pipeline_module  # noqa: E402

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="Run MLOps Pipeline (function-decorators)")
    parser.add_argument(
        "--max-records",
        type=int,
        default=50000,
        help="Maximum number of records to load for training",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["remote", "local", "debug"],
        default="remote",
        help="Pipeline execution mode: remote (default), local, or debug",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("===============================================")
        logger.info("🚀 Starting MLOps Pipeline (function-decorators)")
        logger.info("===============================================")
        logger.info(f"Max records: {args.max_records}")
        logger.info(f"Mode: {args.mode}")

        # Configure controller execution mode
        if args.mode == "debug":
            PipelineDecorator.debug_pipeline()
            logger.info("Running in DEBUG mode (all steps executed synchronously)")
        elif args.mode == "local":
            PipelineDecorator.run_locally()
            logger.info("Running in LOCAL mode (controller local, components as subprocesses)")
        else:
            logger.info("Running in REMOTE mode (controller on services queue)")

        # Kick off the pipeline
        pipeline_module.main(max_records=args.max_records)

        logger.info("===============================================")
        logger.info("🎉 Pipeline invocation completed")
        logger.info("===============================================")
        return 0

    except Exception as e:
        logger.error(f"Pipeline invocation failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
