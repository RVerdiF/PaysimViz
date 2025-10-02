import logging
import os
from pathlib import Path

def setup_logger():
    """Sets up a logger for the application."""
    
    log_dir = Path("logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / "app.log"

    logger = logging.getLogger("PaySimApp")
    logger.setLevel(logging.INFO)

    # Prevent logs from being propagated to the root logger
    logger.propagate = False

    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add handler to logger, but only if it doesn't have handlers already
    if not logger.handlers:
        logger.addHandler(fh)

    return logger
