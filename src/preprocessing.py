"""
===========================================================
Preprocessing Module
===========================================================

This module is responsible for:

1. Loading datasets
2. Cleaning raw text
3. Validating data
4. Creating labels
5. Splitting the dataset

Author : Mahmoud Morsy
Project: Fake News Detection
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================

import re
import string
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config
from logger import get_logger

logger = get_logger(__name__)

# ===========================================================
# Text Cleaning
# ===========================================================

def clean_text(text: str) -> str:
    """
    Clean raw news text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Cleaned text.
    """

    if pd.isna(text):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ===========================================================
# Validate Dataset
# ===========================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate dataset before training.

    Raises
    ------
    ValueError
    """

    logger.info("Validating dataset...")

    if df.empty:
        raise ValueError("Dataset is empty.")

    required_columns = ["text", "label"]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    logger.info("Dataset validation passed.")


# ===========================================================
# Load Dataset
# ===========================================================

def load_dataset() -> pd.DataFrame:
    """
    Load Fake.csv and True.csv.

    Returns
    -------
    Combined DataFrame
    """

    logger.info("=" * 60)
    logger.info("Loading Dataset")
    logger.info("=" * 60)

    fake = pd.read_csv(
        Config.DATA_DIR / "Fake.csv"
    )

    real = pd.read_csv(
        Config.DATA_DIR / "True.csv"
    )

    logger.info(
        f"Fake News : {len(fake):,}"
    )

    logger.info(
        f"Real News : {len(real):,}"
    )

    fake["label"] = 0

    real["label"] = 1

    df = pd.concat(

        [fake, real],

        ignore_index=True

    )

    logger.info(
        f"Total Samples : {len(df):,}"
    )

    return df


# ===========================================================
# Clean Dataset
# ===========================================================

def preprocess_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply preprocessing pipeline.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
    """

    logger.info("=" * 60)
    logger.info("Cleaning Dataset")
    logger.info("=" * 60)

    df = df.copy()

    # Remove missing rows

    df = df.dropna(
        subset=["text"]
    )

    # Remove duplicated articles

    duplicates = df.duplicated().sum()

    if duplicates > 0:

        logger.info(
            f"Removing {duplicates} duplicate rows."
        )

        df = df.drop_duplicates()

    # Clean text

    df["text"] = (
        df["text"]
        .astype(str)
        .apply(clean_text)
    )

    logger.info("Cleaning completed.")

    return df


# ===========================================================
# Dataset Statistics
# ===========================================================

def dataset_statistics(
    df: pd.DataFrame
):
    """
    Display basic dataset statistics.
    """

    logger.info("=" * 60)
    logger.info("Dataset Statistics")
    logger.info("=" * 60)

    logger.info(
        f"Rows    : {len(df):,}"
    )

    logger.info(
        f"Columns : {len(df.columns)}"
    )

    logger.info(
        f"Fake News : {(df['label']==0).sum():,}"
    )

    logger.info(
        f"Real News : {(df['label']==1).sum():,}"
    )

    logger.info(
        f"Missing Values : {df.isna().sum().sum():,}"
    )


# ===========================================================
# Train/Test Split
# ===========================================================

def split_dataset(
    df: pd.DataFrame
) -> Tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series
]:
    """
    Split dataset.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    logger.info("=" * 60)
    logger.info("Splitting Dataset")
    logger.info("=" * 60)

    X = df["text"]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=Config.TEST_SIZE,

        random_state=Config.RANDOM_STATE,

        stratify=y

    )

    logger.info(
        f"Training Samples : {len(X_train):,}"
    )

    logger.info(
        f"Testing Samples  : {len(X_test):,}"
    )

    return (

        X_train,

        X_test,

        y_train,

        y_test

    )


# ===========================================================
# Complete Preprocessing Pipeline
# ===========================================================

def prepare_data():
    """
    Complete preprocessing workflow.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    df = load_dataset()

    df = preprocess_dataset(df)

    validate_dataset(df)

    dataset_statistics(df)

    return split_dataset(df)