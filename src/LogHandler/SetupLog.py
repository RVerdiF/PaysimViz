import logging

def setup_logger():
    """Sets up a logger to output to the console."""
    
    logger = logging.getLogger("PaySimApp")
    logger.setLevel(logging.INFO)

    # Prevent logs from being propagated to the root logger
    logger.propagate = False

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add handler to logger, but only if it doesn't have handlers already
    if not logger.handlers:
        logger.addHandler(ch)

    return logger
