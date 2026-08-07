"""
===========================================================
Artifacts Module
===========================================================

This module is responsible for:

1. Saving trained models
2. Saving vectorizers
3. Saving feature selectors
4. Saving complete pipeline
5. Saving metadata
6. Saving configuration
7. Saving experiment logs
8. Saving selected feature names
9. Saving label mapping
10. Saving feature importance
11. Loading artifacts
12. Verifying saved files

Author : Mahmoud Morsy
Project: Fake News Detection
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import Config
from logger import get_logger

logger = get_logger(__name__)

# ===========================================================
# Save Model
# ===========================================================

def save_model(model):
    """
    Save trained classifier.
    """

    path = Config.ARTIFACT_DIR / Config.MODEL_FILE

    joblib.dump(model, path)

    logger.info(f"✔ Model saved -> {path}")


# ===========================================================
# Save TF-IDF
# ===========================================================

def save_vectorizer(vectorizer):
    """
    Save trained TF-IDF vectorizer.
    """

    path = Config.ARTIFACT_DIR / Config.TFIDF_FILE

    joblib.dump(vectorizer, path)

    logger.info(f"✔ TF-IDF saved -> {path}")


# ===========================================================
# Save Feature Selector
# ===========================================================

def save_selector(selector):
    """
    Save SelectKBest object.
    """

    path = Config.ARTIFACT_DIR / Config.SELECTOR_FILE

    joblib.dump(selector, path)

    logger.info(f"✔ Selector saved -> {path}")


# ===========================================================
# Save Pipeline
# ===========================================================

def save_pipeline(pipeline):
    """
    Save complete Scikit-Learn pipeline.
    """

    path = Config.ARTIFACT_DIR / Config.PIPELINE_FILE

    joblib.dump(pipeline, path)

    logger.info(f"✔ Pipeline saved -> {path}")


# ===========================================================
# Save Metadata
# ===========================================================

def save_metadata(metadata: dict):
    """
    Save project metadata.
    """

    metadata["created_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    path = Config.ARTIFACT_DIR / Config.METADATA_FILE

    with open(path, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    logger.info(f"✔ Metadata saved -> {path}")


# ===========================================================
# Save Config
# ===========================================================

def save_config():
    """
    Save training configuration.
    """

    config = {

        "random_state": Config.RANDOM_STATE,

        "test_size": Config.TEST_SIZE,

        "ngram_range": Config.NGRAM_RANGE,

        "stop_words": Config.STOP_WORDS,

        "selected_features": Config.N_FEATURES,

        "svm_C": Config.SVM_C,

        "loss": Config.LOSS,

        "max_iter": Config.MAX_ITER

    }

    path = Config.ARTIFACT_DIR / Config.CONFIG_FILE

    with open(path, "w") as f:

        json.dump(config, f, indent=4)

    logger.info(f"✔ Config saved -> {path}")


# ===========================================================
# Save Experiment Log
# ===========================================================

def save_experiment(experiment: pd.DataFrame):
    """
    Save experiment history.
    """

    path = Config.ARTIFACT_DIR / Config.EXPERIMENT_FILE

    experiment.to_csv(

        path,

        index=False

    )

    logger.info(f"✔ Experiment log saved -> {path}")


# ===========================================================
# Save Feature Names
# ===========================================================

def save_feature_names(feature_names):
    """
    Save selected feature names.

    Useful for:
        - Explainability
        - SHAP
        - Feature Importance
        - Debugging
    """

    path = Config.ARTIFACT_DIR / "feature_names.npy"

    np.save(path, feature_names)

    logger.info(f"✔ Feature names saved -> {path}")


# ===========================================================
# Save Label Mapping
# ===========================================================

def save_label_mapping():
    """
    Save label mapping.
    """

    labels = {

        "0": "Fake",

        "1": "Real"

    }

    path = Config.ARTIFACT_DIR / "label_mapping.json"

    with open(path, "w") as f:

        json.dump(

            labels,

            f,

            indent=4

        )

    logger.info(f"✔ Label mapping saved -> {path}")


# ===========================================================
# Save Feature Importance
# ===========================================================

def save_feature_importance(

    feature_names,

    selector

):
    """
    Save Chi-Square scores for selected features.
    """

    scores = selector.scores_[

        selector.get_support()

    ]

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Chi2 Score": scores

    })

    importance = importance.sort_values(

        "Chi2 Score",

        ascending=False

    )

    path = Config.ARTIFACT_DIR / "feature_importance.csv"

    importance.to_csv(

        path,

        index=False

    )

    logger.info(f"✔ Feature importance saved -> {path}")


# ===========================================================
# Generic Loader
# ===========================================================

def load_artifact(filename):
    """
    Generic artifact loader.
    """

    path = Config.ARTIFACT_DIR / filename

    return joblib.load(path)


# ===========================================================
# Load Metadata
# ===========================================================

def load_metadata():

    with open(

        Config.ARTIFACT_DIR /

        Config.METADATA_FILE,

        "r"

    ) as f:

        return json.load(f)


# ===========================================================
# Load Config
# ===========================================================

def load_config():

    with open(

        Config.ARTIFACT_DIR /

        Config.CONFIG_FILE,

        "r"

    ) as f:

        return json.load(f)


# ===========================================================
# Load Feature Names
# ===========================================================

def load_feature_names():

    return np.load(

        Config.ARTIFACT_DIR /

        "feature_names.npy",

        allow_pickle=True

    )


# ===========================================================
# Load Label Mapping
# ===========================================================

def load_label_mapping():

    with open(

        Config.ARTIFACT_DIR /

        "label_mapping.json",

        "r"

    ) as f:

        return json.load(f)


# ===========================================================
# Verify Artifacts
# ===========================================================

def verify_artifacts():
    """
    Verify all production files exist.
    """

    required_files = [

        Config.MODEL_FILE,

        Config.PIPELINE_FILE,

        Config.TFIDF_FILE,

        Config.SELECTOR_FILE,

        Config.METADATA_FILE,

        Config.CONFIG_FILE,

        Config.EXPERIMENT_FILE,

        "feature_names.npy",

        "feature_importance.csv",

        "label_mapping.json"

    ]

    logger.info("=" * 60)

    logger.info("Verifying Artifacts")

    logger.info("=" * 60)

    missing = []

    for file in required_files:

        path = Config.ARTIFACT_DIR / file

        if path.exists():

            logger.info(f"✔ {file}")

        else:

            logger.warning(f"✘ Missing: {file}")

            missing.append(file)

    if len(missing) == 0:

        logger.info("✔ All artifacts verified successfully.")

    else:

        logger.warning(f"{len(missing)} missing artifact(s).")

        logger.warning(missing)


# ===========================================================
# Save Everything
# ===========================================================

def save_all_artifacts(

    model,

    vectorizer,

    selector,

    pipeline,

    metadata,

    experiment,

    feature_names

):
    """
    Save every artifact produced during training.
    """

    logger.info("=" * 60)

    logger.info("Saving Production Artifacts")

    logger.info("=" * 60)

    save_model(model)

    save_vectorizer(vectorizer)

    save_selector(selector)

    save_pipeline(pipeline)

    save_metadata(metadata)

    save_config()

    save_experiment(experiment)

    save_feature_names(feature_names)

    save_label_mapping()

    save_feature_importance(

        feature_names,

        selector

    )

    verify_artifacts()

    logger.info("=" * 60)

    logger.info("Production Artifacts Ready")

    logger.info("=" * 60)