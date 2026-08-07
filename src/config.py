"""
===========================================================
Configuration File
===========================================================

This module contains all project-wide configuration values.

Changing parameters here automatically updates the
entire project.

===========================================================
"""

from pathlib import Path


class Config:
    """
    Global configuration for the Fake News Detection project.
    """

    # =======================================================
    # Randomness
    # =======================================================

    RANDOM_STATE = 42

    # =======================================================
    # Dataset
    # =======================================================

    TEST_SIZE = 0.20

    # =======================================================
    # TF-IDF
    # =======================================================

    STOP_WORDS = "english"

    NGRAM_RANGE = (1, 2)

    # =======================================================
    # Feature Selection
    # =======================================================

    N_FEATURES = 10000

    # =======================================================
    # Linear SVM
    # =======================================================

    SVM_C = 10

    MAX_ITER = 10000

    LOSS = "squared_hinge"

    # =======================================================
    # Project Information
    # =======================================================

    PROJECT_NAME = "Fake News Detection"

    MODEL_NAME = "Linear SVM"

    VERSION = "1.0.0"

    AUTHOR = "Mahmoud Morsy"

    # =======================================================
    # Paths
    # =======================================================

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR = PROJECT_ROOT / "data"

    ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

    REPORT_DIR = PROJECT_ROOT / "reports"

    NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

    TEST_DIR = PROJECT_ROOT / "tests"

    LOG_DIR = PROJECT_ROOT / "logs"

    # =======================================================
    # Artifact Names
    # =======================================================

MODEL_FILE = "svm_model.joblib"
PIPELINE_FILE = "pipeline.joblib"
TFIDF_FILE = "tfidf_vectorizer.joblib"
SELECTOR_FILE = "selector.joblib"
METADATA_FILE = "metadata.json"
CONFIG_FILE = "config.json"
EXPERIMENT_FILE = "experiment_log.csv"
FEATURE_NAMES_FILE = "feature_names.npy"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
LABEL_MAPPING_FILE = "label_mapping.json"