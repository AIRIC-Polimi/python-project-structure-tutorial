import logging
import os

def get_logger(level: str = None):
    logger = logging.getLogger("aim_tutorial")

    logHandler = logging.StreamHandler()
    logger.addHandler(logHandler)

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(level)
    return logger
