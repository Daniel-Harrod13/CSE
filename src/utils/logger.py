"""
Logger Utility Module.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup the logger with the specified name, log file, and level.
    
    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
                                  If None, logs only to console.
        level (int, optional): Logging level.
                               Defaults to logging.INFO.
                               
    Returns:
        logging.Logger: Configured logger.
    """
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create a console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create a file handler if log_file is specified
    if log_file:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name="ev_adoption_analysis"):
    """
    Get a logger instance.
    
    Args:
        name (str, optional): Name of the logger.
                              Defaults to "ev_adoption_analysis".
                              
    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger(name) 