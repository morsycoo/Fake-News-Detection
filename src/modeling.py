"""
===========================================================
Modeling Module
===========================================================

This module is responsible for:

1. Building machine learning models
2. Training models
3. Hyperparameter tuning
4. Cross Validation
5. Model Factory

Author : Mahmoud Morsy
Project: Fake News Detection
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================

from dataclasses import dataclass
from time import time
from typing import Any, Dict

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import Config
from logger import get_logger

logger = get_logger(__name__)


# ===========================================================
# Dataclass
# ===========================================================

@dataclass
class ModelArtifacts:
    """
    Store everything related to a trained model.
    """

    model: BaseEstimator

    train_time: float

    best_params: Dict | None = None

    cv_score: float | None = None


# ===========================================================
# Model Factory
# ===========================================================

def build_model(
    model_name: str
) -> BaseEstimator:
    """
    Factory function for creating models.

    Supported Models
    ----------------
    - naive_bayes
    - logistic_regression
    - linear_svm
    """

    logger.info(f"Building {model_name}...")

    model_name = model_name.lower()

    if model_name == "naive_bayes":

        return MultinomialNB()

    elif model_name == "logistic_regression":

        return LogisticRegression(

            max_iter=1000,

            random_state=Config.RANDOM_STATE

        )

    elif model_name == "linear_svm":

        return LinearSVC(

            C=Config.SVM_C,

            loss=Config.LOSS,

            random_state=Config.RANDOM_STATE,

            max_iter=Config.MAX_ITER

        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ===========================================================
# Train Model
# ===========================================================

def train_model(
    model: BaseEstimator,
    X_train,
    y_train
) -> ModelArtifacts:
    """
    Train a machine learning model.
    """

    logger.info("=" * 60)
    logger.info("Training Model")
    logger.info("=" * 60)

    start = time()

    model.fit(
        X_train,
        y_train
    )

    elapsed = round(
        time() - start,
        2
    )

    logger.info(
        f"Training completed in {elapsed} sec."
    )

    return ModelArtifacts(

        model=model,

        train_time=elapsed

    )


# ===========================================================
# Cross Validation
# ===========================================================

def cross_validate_model(
    model: BaseEstimator,
    X_train,
    y_train,
    cv: int = 5
):
    """
    Perform Cross Validation.
    """

    logger.info("Running Cross Validation...")

    scores = cross_val_score(

        model,

        X_train,

        y_train,

        cv=cv,

        scoring="f1",

        n_jobs=-1

    )

    logger.info(
        f"Mean F1 : {scores.mean():.4f}"
    )

    logger.info(
        f"Std F1 : {scores.std():.4f}"
    )

    return scores


# ===========================================================
# Grid Search
# ===========================================================

def grid_search_svm(
    X_train,
    y_train
) -> ModelArtifacts:
    """
    Hyperparameter tuning using GridSearchCV.
    """

    logger.info("=" * 60)
    logger.info("Grid Search")
    logger.info("=" * 60)

    param_grid = {

        "C": [0.1, 1, 10, 100]

    }

    model = LinearSVC(

        random_state=Config.RANDOM_STATE,

        max_iter=Config.MAX_ITER

    )

    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        scoring="f1",

        cv=5,

        n_jobs=-1

    )

    start = time()

    grid.fit(
        X_train,
        y_train
    )

    elapsed = round(
        time() - start,
        2
    )

    logger.info(
        f"Best Parameters : {grid.best_params_}"
    )

    logger.info(
        f"Best Score : {grid.best_score_:.4f}"
    )

    return ModelArtifacts(

        model=grid.best_estimator_,

        train_time=elapsed,

        best_params=grid.best_params_,

        cv_score=grid.best_score_

    )


# ===========================================================
# Random Search
# ===========================================================

def random_search_svm(
    X_train,
    y_train
) -> ModelArtifacts:
    """
    Hyperparameter tuning using RandomizedSearchCV.
    """

    logger.info("=" * 60)
    logger.info("Random Search")
    logger.info("=" * 60)

    param_dist = {

        "C": np.logspace(-2, 2, 20)

    }

    model = LinearSVC(

        random_state=Config.RANDOM_STATE,

        max_iter=Config.MAX_ITER

    )

    random_search = RandomizedSearchCV(

        estimator=model,

        param_distributions=param_dist,

        n_iter=10,

        cv=5,

        scoring="f1",

        random_state=Config.RANDOM_STATE,

        n_jobs=-1

    )

    start = time()

    random_search.fit(

        X_train,

        y_train

    )

    elapsed = round(
        time() - start,
        2
    )

    logger.info(
        f"Best Parameters : {random_search.best_params_}"
    )

    logger.info(
        f"Best Score : {random_search.best_score_:.4f}"
    )

    return ModelArtifacts(

        model=random_search.best_estimator_,

        train_time=elapsed,

        best_params=random_search.best_params_,

        cv_score=random_search.best_score_

    )


# ===========================================================
# Model Summary
# ===========================================================

def summarize_model(
    artifacts: ModelArtifacts
):
    """
    Print a concise summary of a trained model.
    """

    logger.info("=" * 60)
    logger.info("Model Summary")
    logger.info("=" * 60)

    logger.info(
        f"Model : {type(artifacts.model).__name__}"
    )

    logger.info(
        f"Training Time : {artifacts.train_time} sec"
    )

    if artifacts.best_params:

        logger.info(
            f"Best Params : {artifacts.best_params}"
        )

    if artifacts.cv_score:

        logger.info(
            f"CV Score : {artifacts.cv_score:.4f}"
        )