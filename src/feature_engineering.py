"""
===========================================================
Feature Engineering Module
===========================================================

This module is responsible for:

1. TF-IDF Vectorization
2. Feature Selection
3. Vocabulary Statistics
4. Feature Extraction

Author : Mahmoud Morsy
Project: Fake News Detection
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from config import Config
from logger import get_logger

logger = get_logger(__name__)

# ===========================================================
# Build TF-IDF Vectorizer
# ===========================================================

def build_vectorizer() -> TfidfVectorizer:
    """
    Create TF-IDF Vectorizer.

    Returns
    -------
    TfidfVectorizer
    """

    logger.info("=" * 60)
    logger.info("Building TF-IDF Vectorizer")
    logger.info("=" * 60)

    vectorizer = TfidfVectorizer(

        stop_words=Config.STOP_WORDS,

        ngram_range=Config.NGRAM_RANGE,

        lowercase=True,

        sublinear_tf=True

    )

    return vectorizer


# ===========================================================
# Build Feature Selector
# ===========================================================

def build_selector() -> SelectKBest:
    """
    Create Chi-Square Feature Selector.

    Returns
    -------
    SelectKBest
    """

    logger.info("Building SelectKBest...")

    selector = SelectKBest(

        score_func=chi2,

        k=Config.N_FEATURES

    )

    return selector


# ===========================================================
# Fit TF-IDF
# ===========================================================

def fit_vectorizer(
    vectorizer: TfidfVectorizer,
    X_train: pd.Series
):
    """
    Fit TF-IDF using only training data.
    """

    logger.info("Fitting TF-IDF...")

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    logger.info(
        f"Vocabulary Size : {len(vectorizer.vocabulary_):,}"
    )

    logger.info(
        f"TF-IDF Shape : {X_train_tfidf.shape}"
    )

    return X_train_tfidf


# ===========================================================
# Transform Dataset
# ===========================================================

def transform_dataset(
    vectorizer: TfidfVectorizer,
    X
):
    """
    Transform any dataset using
    fitted TF-IDF.
    """

    return vectorizer.transform(X)


# ===========================================================
# Select Best Features
# ===========================================================

def fit_selector(
    selector: SelectKBest,
    X_train,
    y_train
):
    """
    Select top K features.
    """

    logger.info("Selecting Best Features...")

    X_selected = selector.fit_transform(

        X_train,

        y_train

    )

    logger.info(
        f"Selected Features : {X_selected.shape[1]}"
    )

    return X_selected


# ===========================================================
# Transform Features
# ===========================================================

def transform_features(
    selector,
    X
):
    """
    Apply trained selector.
    """

    return selector.transform(X)


# ===========================================================
# Selected Feature Names
# ===========================================================

def get_selected_feature_names(
    vectorizer,
    selector
):
    """
    Return selected feature names.
    """

    feature_names = np.array(

        vectorizer.get_feature_names_out()

    )

    selected_names = feature_names[
        selector.get_support()
    ]

    return selected_names


# ===========================================================
# Vocabulary Statistics
# ===========================================================

def vocabulary_statistics(
    vectorizer
):
    """
    Print vocabulary information.
    """

    logger.info("=" * 60)
    logger.info("Vocabulary Statistics")
    logger.info("=" * 60)

    logger.info(
        f"Vocabulary Size : {len(vectorizer.vocabulary_):,}"
    )

    sample = list(
        vectorizer.vocabulary_.keys()
    )[:20]

    logger.info(
        f"Sample Words : {sample}"
    )


# ===========================================================
# Feature Statistics
# ===========================================================

def feature_statistics(
    selector
):
    """
    Display feature selection statistics.
    """

    logger.info("=" * 60)
    logger.info("Feature Selection Statistics")
    logger.info("=" * 60)

    logger.info(
        f"Selected Features : {selector.get_support().sum():,}"
    )


# ===========================================================
# Complete Feature Pipeline
# ===========================================================

def prepare_features(
    X_train,
    X_test,
    y_train
) -> Dict:
    """
    Complete feature engineering pipeline.

    Returns
    -------
    Dictionary
    """

    vectorizer = build_vectorizer()

    selector = build_selector()

    X_train_tfidf = fit_vectorizer(

        vectorizer,

        X_train

    )

    X_test_tfidf = transform_dataset(

        vectorizer,

        X_test

    )

    X_train_selected = fit_selector(

        selector,

        X_train_tfidf,

        y_train

    )

    X_test_selected = transform_features(

        selector,

        X_test_tfidf

    )

    vocabulary_statistics(
        vectorizer
    )

    feature_statistics(
        selector
    )

    selected_words = get_selected_feature_names(

        vectorizer,

        selector

    )

    logger.info(
        f"Returned Selected Words : {len(selected_words):,}"
    )

    return {

        "vectorizer": vectorizer,

        "selector": selector,

        "selected_words": selected_words,

        "X_train": X_train_selected,

        "X_test": X_test_selected

    }