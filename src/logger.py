"""
===========================================================
Logger Configuration
===========================================================
"""

import logging
import sys

from config import Config


def get_logger(name: str) -> logging.Logger:
    """
    Create a project logger.

    Parameters
    ----------
    name : str

    Returns
    -------
    logging.Logger
    """

    Config.LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(message)s"

    )

    # Console Logger

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File Logger

    file_handler = logging.FileHandler(

        Config.LOG_DIR / "training.log"

    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger