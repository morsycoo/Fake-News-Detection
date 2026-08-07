"""
===========================================================
Utility Functions
===========================================================
"""

import json
import time
from pathlib import Path


def timer(start_time: float) -> float:
    """
    Return elapsed time in seconds.
    """

    return round(time.time() - start_time, 2)


def create_directory(path: Path):
    """
    Create directory if it does not exist.
    """

    path.mkdir(

        parents=True,

        exist_ok=True

    )


def save_json(
    data: dict,
    filepath: Path
):
    """
    Save dictionary as JSON.
    """

    with open(

        filepath,

        "w"

    ) as f:

        json.dump(

            data,

            f,

            indent=4

        )


def load_json(
    filepath: Path
):
    """
    Load JSON file.
    """

    with open(

        filepath,

        "r"

    ) as f:

        return json.load(f)