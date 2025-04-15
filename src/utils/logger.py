"""
Logging utilities for the Dyadic Interaction Dataset Generator.
"""

import logging
import os
import sys
import coloredlogs
from typing import Dict, Any, Optional

def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """
    Set up and configure the logger.
    
    Args:
        config: Configuration dictionary with logging settings
            - log_level: Logging level (debug, info, warning, error, critical)
            - log_file: Optional path to log file
            
    Returns:
        Configured logger instance
    """
    # Extract settings
    log_level_str = config.get('general', {}).get('log_level', 'info').upper()
    log_file = config.get('general', {}).get('log_file', None)
    debug_mode = config.get('general', {}).get('debug', False)
    
    # Map string log level to actual log level
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Set debug level if debug mode is enabled
    if debug_mode:
        log_level = logging.DEBUG
    
    # Create logger
    logger = logging.getLogger('dyadic_interaction')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Install colored logs
    coloredlogs.install(
        level=log_level,
        logger=logger,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Logger initialized with level {log_level_str}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    return logger

def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    Log the configuration settings.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Configuration settings:")
    
    # Log each section
    for section, settings in config.items():
        logger.info(f"  {section}:")
        
        if isinstance(settings, dict):
            for key, value in settings.items():
                logger.info(f"    {key}: {value}")
        else:
            logger.info(f"    {settings}")
    
    logger.info("Configuration loaded successfully") 