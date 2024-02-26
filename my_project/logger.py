import logging
import os


def get_logger(level: str = None, filename: str = None):
    logger = logging.getLogger("aim_tutorial")

    logHandler = logging.StreamHandler()
    logger.addHandler(logHandler)

    if filename is not None:
        logger.addHandler(logging.FileHandler(filename, "w"))

    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    logger.setLevel(level)
    return logger
