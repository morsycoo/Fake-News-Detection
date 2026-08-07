"""
=============================================================
Fake News Detection - Training Pipeline
Author: Mahmoud Morsy
Project: Fake News Detection using Classical NLP
Model: Linear Support Vector Machine (LinearSVC)
=============================================================

This script is responsible for:

1. Loading the dataset.
2. Cleaning text.
3. Building the NLP pipeline.
4. Training the model.
5. Evaluating performance.
6. Saving production artifacts.

=============================================================
"""

# ============================================================
# Standard Library Imports
# ============================================================

import json
import logging
import re
import string
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# ============================================================
# Third-Party Imports
# ============================================================

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# ============================================================
# Configure Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ============================================================
# Project Configuration
# ============================================================

class Config:
    """
    Central configuration for the entire training pipeline.

    Any parameter that may change later should be defined here.
    """

    RANDOM_STATE = 42

    TEST_SIZE = 0.20

    N_FEATURES = 10000

    NGRAM_RANGE = (1, 2)

    STOP_WORDS = "english"

    SVM_C = 10

    MAX_ITER = 10000

    MODEL_NAME = "Linear SVM"

# ============================================================
# Project Directories
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

REPORT_DIR = PROJECT_ROOT / "reports"

# Create directories automatically if they do not exist
ARTIFACT_DIR.mkdir(exist_ok=True)

REPORT_DIR.mkdir(exist_ok=True)

# ============================================================
# Utility Functions
# ============================================================

def timer(start_time: float) -> float:
    """
    Return elapsed execution time in seconds.
    """

    return round(time.time() - start_time, 2)


# ============================================================
# Text Cleaning Function
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean raw news text before vectorization.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation
    5. Remove numbers
    6. Remove extra spaces
    """

    if pd.isna(text):
        return ""

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# Load Dataset
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load Fake.csv and True.csv then combine them.

    Returns
    -------
    DataFrame

    Columns

    text
    label
    """

    logger.info("Loading datasets...")

    fake = pd.read_csv(
        DATA_DIR / "Fake.csv"
    )

    real = pd.read_csv(
        DATA_DIR / "True.csv"
    )

    fake["label"] = 0

    real["label"] = 1

    df = pd.concat(
        [fake, real],
        ignore_index=True
    )

    logger.info(
        "Dataset loaded successfully."
    )

    logger.info(
        f"Total Samples: {len(df):,}"
    )

    return df


# ============================================================
# Preprocess Dataset
# ============================================================

def preprocess_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean all news articles.
    """

    logger.info("Cleaning text...")

    df["text"] = (
        df["text"]
        .astype(str)
        .apply(clean_text)
    )

    logger.info("Cleaning completed.")

    return df


# ============================================================
# Train / Test Split
# ============================================================

def split_dataset(
    df: pd.DataFrame
) -> Tuple:

    logger.info("Splitting dataset...")

    X = df["text"]

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=Config.TEST_SIZE,

        stratify=y,

        random_state=Config.RANDOM_STATE

    )

    logger.info(
        f"Training Samples : {len(X_train):,}"
    )

    logger.info(
        f"Testing Samples  : {len(X_test):,}"
    )

    return X_train, X_test, y_train, y_test

# ============================================================
# Feature Engineering
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

# ============================================================
# Machine Learning Model
# ============================================================

from sklearn.svm import LinearSVC


# ============================================================
# Create TF-IDF Vectorizer
# ============================================================

def build_vectorizer() -> TfidfVectorizer:
    """
    Create and return a TF-IDF Vectorizer.

    Why TF-IDF?
    ------------
    TF-IDF reduces the importance of very common words while
    increasing the weight of informative words.

    Parameters are centralized inside the Config class to make
    future experiments easier.
    """

    logger.info("Building TF-IDF Vectorizer...")

    vectorizer = TfidfVectorizer(

        stop_words=Config.STOP_WORDS,

        ngram_range=Config.NGRAM_RANGE,

        lowercase=True,

        sublinear_tf=True

    )

    return vectorizer


# ============================================================
# Create Feature Selector
# ============================================================

def build_feature_selector() -> SelectKBest:
    """
    Select the most informative features using Chi-Square.

    Why Feature Selection?
    ----------------------
    TF-IDF may generate hundreds of thousands of features.

    Keeping only the most informative ones:

    - reduces memory usage
    - speeds up training
    - reduces noise
    - improves generalization
    """

    logger.info("Building Feature Selector...")

    selector = SelectKBest(

        score_func=chi2,

        k=Config.N_FEATURES

    )

    return selector


# ============================================================
# Create Linear SVM Classifier
# ============================================================

def build_classifier() -> LinearSVC:
    """
    Create the final production model.

    Best Model:
        Linear SVM

    Hyperparameters were selected using
    GridSearchCV + Cross Validation.
    """

    logger.info("Building Linear SVM...")

    model = LinearSVC(

        C=Config.SVM_C,

        random_state=Config.RANDOM_STATE,

        max_iter=Config.MAX_ITER,

        loss="squared_hinge"

    )

    return model


# ============================================================
# Train Complete NLP Pipeline
# ============================================================

def train_pipeline(
    X_train: pd.Series,
    y_train: pd.Series
):
    """
    Build and train the entire NLP pipeline.

    Pipeline Steps
    --------------

    Raw Text

        ↓

    TF-IDF Vectorization

        ↓

    SelectKBest

        ↓

    Linear SVM

    Returns
    -------
    pipeline
    """

    logger.info("=" * 60)
    logger.info("Training Pipeline...")
    logger.info("=" * 60)

    start = time.time()

    vectorizer = build_vectorizer()

    selector = build_feature_selector()

    classifier = build_classifier()

    logger.info("Fitting TF-IDF...")

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    logger.info(
        f"TF-IDF Features : {X_train_tfidf.shape[1]:,}"
    )

    logger.info("Selecting Best Features...")

    X_train_selected = selector.fit_transform(

        X_train_tfidf,

        y_train

    )

    logger.info(
        f"Selected Features : {X_train_selected.shape[1]:,}"
    )

    logger.info("Training Linear SVM...")

    classifier.fit(

        X_train_selected,

        y_train

    )

    elapsed = timer(start)

    logger.info(
        f"Training completed in {elapsed} seconds."
    )

    pipeline = {

        "vectorizer": vectorizer,

        "selector": selector,

        "classifier": classifier

    }

    return pipeline


# ============================================================
# Transform Test Data
# ============================================================

def transform_test_data(
    pipeline: Dict,
    X_test: pd.Series
):
    """
    Apply the exact same preprocessing steps
    used during training.

    IMPORTANT
    ---------
    Never fit TF-IDF or Feature Selection
    on the test data.

    Doing so would cause Data Leakage.
    """

    logger.info("Transforming Test Data...")

    X_test_tfidf = pipeline["vectorizer"].transform(
        X_test
    )

    X_test_selected = pipeline["selector"].transform(
        X_test_tfidf
    )

    return X_test_selected

# ============================================================
# Model Evaluation
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(
    pipeline: Dict,
    X_test_selected,
    y_test: pd.Series,
) -> Dict:
    """
    Evaluate the trained model on the test dataset.

    Returns
    -------
    Dictionary containing:

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - PR-AUC
    - Predictions
    - Decision Scores
    """

    logger.info("=" * 60)
    logger.info("Evaluating Model...")
    logger.info("=" * 60)

    classifier = pipeline["classifier"]

    # --------------------------------------------------------
    # Generate Predictions
    # --------------------------------------------------------

    y_pred = classifier.predict(X_test_selected)

    # --------------------------------------------------------
    # Decision Scores
    #
    # LinearSVC does not provide probabilities.
    # Instead, we use decision_function().
    # --------------------------------------------------------

    decision_scores = classifier.decision_function(
        X_test_selected
    )

    # --------------------------------------------------------
    # Compute Metrics
    # --------------------------------------------------------

    results = {

        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "precision": precision_score(
            y_test,
            y_pred
        ),

        "recall": recall_score(
            y_test,
            y_pred
        ),

        "f1": f1_score(
            y_test,
            y_pred
        ),

        "roc_auc": roc_auc_score(
            y_test,
            decision_scores
        ),

        "pr_auc": average_precision_score(
            y_test,
            decision_scores
        ),

        "predictions": y_pred,

        "decision_scores": decision_scores,

    }

    logger.info("Evaluation completed.")

    return results


# ============================================================
# Print Evaluation Report
# ============================================================

def print_metrics(results: Dict):
    """
    Print evaluation metrics in a readable format.
    """

    logger.info("=" * 60)
    logger.info("Evaluation Report")
    logger.info("=" * 60)

    logger.info(
        f"Accuracy : {results['accuracy']:.4f}"
    )

    logger.info(
        f"Precision: {results['precision']:.4f}"
    )

    logger.info(
        f"Recall   : {results['recall']:.4f}"
    )

    logger.info(
        f"F1 Score : {results['f1']:.4f}"
    )

    logger.info(
        f"ROC-AUC  : {results['roc_auc']:.4f}"
    )

    logger.info(
        f"PR-AUC   : {results['pr_auc']:.4f}"
    )


# ============================================================
# Classification Report
# ============================================================

def generate_classification_report(
    y_test,
    results
):
    """
    Generate a full classification report.

    This report includes:

    - Precision
    - Recall
    - F1
    - Support
    """

    report = classification_report(

        y_test,

        results["predictions"],

        digits=4

    )

    logger.info("\nClassification Report\n")

    logger.info(report)

    return report


# ============================================================
# Confusion Matrix
# ============================================================

def generate_confusion_matrix(
    y_test,
    results
):
    """
    Compute confusion matrix.

    Layout

            Predicted

              0     1

    True 0   TN    FP

    True 1   FN    TP
    """

    cm = confusion_matrix(

        y_test,

        results["predictions"]

    )

    logger.info("\nConfusion Matrix")

    logger.info(f"\n{cm}")

    return cm


# ============================================================
# Save Evaluation Report
# ============================================================

def save_evaluation_report(
    results: Dict
):
    """
    Save evaluation metrics as CSV.

    This file can later be used for:

    - Benchmarking
    - Experiment Tracking
    - Dashboarding
    """

    report = pd.DataFrame({

        "Metric":[

            "Accuracy",

            "Precision",

            "Recall",

            "F1",

            "ROC-AUC",

            "PR-AUC"

        ],

        "Value":[

            results["accuracy"],

            results["precision"],

            results["recall"],

            results["f1"],

            results["roc_auc"],

            results["pr_auc"]

        ]

    })

    report.to_csv(

        REPORT_DIR / "evaluation_report.csv",

        index=False

    )

    logger.info(
        "Evaluation report saved."
    )


# ============================================================
# Model Benchmark Summary
# ============================================================

def benchmark_summary(
    results: Dict
):
    """
    Display a compact benchmark summary.

    This function is useful for comparing
    multiple trained models.
    """

    benchmark = pd.DataFrame({

        "Model":[
            Config.MODEL_NAME
        ],

        "Accuracy":[
            results["accuracy"]
        ],

        "Precision":[
            results["precision"]
        ],

        "Recall":[
            results["recall"]
        ],

        "F1":[
            results["f1"]
        ],

        "ROC-AUC":[
            results["roc_auc"]
        ],

        "PR-AUC":[
            results["pr_auc"]
        ]

    })

    logger.info("\nBenchmark Summary")

    logger.info(f"\n{benchmark}")

    benchmark.to_csv(

        REPORT_DIR / "benchmark.csv",

        index=False

    )

    return benchmark

# ============================================================
# Save Production Artifacts
# ============================================================

from datetime import datetime


# ============================================================
# Save Trained Pipeline Components
# ============================================================

def save_artifacts(
    pipeline: Dict
):
    """
    Save all trained artifacts.

    Artifacts are the objects required later during inference.

    Saved Files
    -----------
    - svm_model.joblib
    - tfidf_vectorizer.joblib
    - selector.joblib
    - pipeline.joblib
    """

    logger.info("=" * 60)
    logger.info("Saving Artifacts...")
    logger.info("=" * 60)

    # --------------------------------------------------------
    # Save Classifier
    # --------------------------------------------------------

    joblib.dump(

        pipeline["classifier"],

        ARTIFACT_DIR / "svm_model.joblib"

    )

    logger.info("✔ svm_model.joblib saved.")

    # --------------------------------------------------------
    # Save TF-IDF
    # --------------------------------------------------------

    joblib.dump(

        pipeline["vectorizer"],

        ARTIFACT_DIR / "tfidf_vectorizer.joblib"

    )

    logger.info("✔ tfidf_vectorizer.joblib saved.")

    # --------------------------------------------------------
    # Save Feature Selector
    # --------------------------------------------------------

    joblib.dump(

        pipeline["selector"],

        ARTIFACT_DIR / "selector.joblib"

    )

    logger.info("✔ selector.joblib saved.")

    # --------------------------------------------------------
    # Save Complete Pipeline
    # --------------------------------------------------------

    full_pipeline = Pipeline([

        ("vectorizer", pipeline["vectorizer"]),

        ("selector", pipeline["selector"]),

        ("classifier", pipeline["classifier"])

    ])

    joblib.dump(

        full_pipeline,

        ARTIFACT_DIR / "pipeline.joblib"

    )

    logger.info("✔ pipeline.joblib saved.")


# ============================================================
# Save Metadata
# ============================================================

def save_metadata(
    results: Dict,
    X_train,
    X_test
):
    """
    Save model metadata.

    Metadata describes the trained model and
    can later be displayed inside APIs or dashboards.
    """

    logger.info("Saving metadata...")

    metadata = {

        "project": "Fake News Detection",

        "model": Config.MODEL_NAME,

        "algorithm": "LinearSVC",

        "vectorizer": "TF-IDF",

        "feature_selector": "SelectKBest",

        "selected_features": Config.N_FEATURES,

        "ngram_range": Config.NGRAM_RANGE,

        "training_samples": len(X_train),

        "testing_samples": len(X_test),

        "accuracy": round(results["accuracy"], 5),

        "precision": round(results["precision"], 5),

        "recall": round(results["recall"], 5),

        "f1": round(results["f1"], 5),

        "roc_auc": round(results["roc_auc"], 5),

        "pr_auc": round(results["pr_auc"], 5),

        "training_date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    }

    with open(

        ARTIFACT_DIR / "metadata.json",

        "w"

    ) as f:

        json.dump(

            metadata,

            f,

            indent=4

        )

    logger.info("✔ metadata.json saved.")


# ============================================================
# Save Configuration
# ============================================================

def save_config():
    """
    Save project configuration.

    This allows future training to reproduce
    exactly the same settings.
    """

    logger.info("Saving config...")

    config = {

        "random_state": Config.RANDOM_STATE,

        "test_size": Config.TEST_SIZE,

        "selected_features": Config.N_FEATURES,

        "ngram_range": Config.NGRAM_RANGE,

        "stop_words": Config.STOP_WORDS,

        "svm_c": Config.SVM_C,

        "max_iter": Config.MAX_ITER

    }

    with open(

        ARTIFACT_DIR / "config.json",

        "w"

    ) as f:

        json.dump(

            config,

            f,

            indent=4

        )

    logger.info("✔ config.json saved.")


# ============================================================
# Save Experiment Log
# ============================================================

def save_experiment_log(
    results: Dict
):
    """
    Save experiment history.

    Every training session should create
    a reproducible experiment record.
    """

    logger.info("Saving experiment log...")

    experiment = pd.DataFrame({

        "Experiment ID": [

            datetime.now().strftime(
                "EXP-%Y%m%d-%H%M%S"
            )

        ],

        "Model": [

            Config.MODEL_NAME

        ],

        "Features": [

            Config.N_FEATURES

        ],

        "Accuracy": [

            results["accuracy"]

        ],

        "Precision": [

            results["precision"]

        ],

        "Recall": [

            results["recall"]

        ],

        "F1": [

            results["f1"]

        ],

        "ROC-AUC": [

            results["roc_auc"]

        ],

        "PR-AUC": [

            results["pr_auc"]

        ],

        "Status": [

            "Production"

        ]

    })

    experiment.to_csv(

        ARTIFACT_DIR / "experiment_log.csv",

        index=False

    )

    logger.info("✔ experiment_log.csv saved.")


# ============================================================
# Verify Saved Files
# ============================================================

def verify_artifacts():
    """
    Verify that every required artifact exists.

    This acts as a safety check before deployment.
    """

    logger.info("=" * 60)
    logger.info("Verifying Saved Files...")
    logger.info("=" * 60)

    required_files = [

        "svm_model.joblib",

        "tfidf_vectorizer.joblib",

        "selector.joblib",

        "pipeline.joblib",

        "metadata.json",

        "config.json",

        "experiment_log.csv"

    ]

    missing = []

    for file in required_files:

        path = ARTIFACT_DIR / file

        if path.exists():

            logger.info(f"✔ {file}")

        else:

            logger.error(f"✘ {file}")

            missing.append(file)

    if len(missing) == 0:

        logger.info("\nAll production artifacts verified successfully.")

    else:

        logger.warning("\nSome artifacts are missing.")

        logger.warning(missing)

# ============================================================
# Main Training Pipeline
# ============================================================

def main():
    """
    Main training workflow.

    This function orchestrates the complete machine learning
    pipeline from loading the dataset to saving production
    artifacts.

    Workflow
    --------
    1. Load Dataset
    2. Clean Dataset
    3. Split Dataset
    4. Train NLP Pipeline
    5. Transform Test Set
    6. Evaluate Model
    7. Save Reports
    8. Save Artifacts
    9. Verify Artifacts
    """

    logger.info("=" * 70)
    logger.info("Fake News Detection Training Pipeline Started")
    logger.info("=" * 70)

    total_start = time.time()

    try:

        # ----------------------------------------------------
        # Load Dataset
        # ----------------------------------------------------

        df = load_dataset()

        # ----------------------------------------------------
        # Clean Dataset
        # ----------------------------------------------------

        df = preprocess_dataset(df)

        # ----------------------------------------------------
        # Train/Test Split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = split_dataset(df)

        # ----------------------------------------------------
        # Train Pipeline
        # ----------------------------------------------------

        pipeline = train_pipeline(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Transform Test Data
        # ----------------------------------------------------

        X_test_selected = transform_test_data(
            pipeline,
            X_test
        )

        # ----------------------------------------------------
        # Evaluate Model
        # ----------------------------------------------------

        results = evaluate_model(
            pipeline,
            X_test_selected,
            y_test
        )

        # ----------------------------------------------------
        # Display Metrics
        # ----------------------------------------------------

        print_metrics(results)

        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        generate_classification_report(
            y_test,
            results
        )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        generate_confusion_matrix(
            y_test,
            results
        )

        # ----------------------------------------------------
        # Save Evaluation Reports
        # ----------------------------------------------------

        save_evaluation_report(results)

        benchmark_summary(results)

        # ----------------------------------------------------
        # Save Production Files
        # ----------------------------------------------------

        save_artifacts(pipeline)

        save_metadata(
            results,
            X_train,
            X_test
        )

        save_config()

        save_experiment_log(results)

        # ----------------------------------------------------
        # Verify Saved Files
        # ----------------------------------------------------

        verify_artifacts()

        # ----------------------------------------------------
        # Total Training Time
        # ----------------------------------------------------

        total_time = timer(total_start)

        logger.info("=" * 70)
        logger.info("Training Finished Successfully")
        logger.info("=" * 70)

        logger.info(f"Total Time : {total_time} sec")

        logger.info(f"Accuracy   : {results['accuracy']:.4f}")
        logger.info(f"Precision  : {results['precision']:.4f}")
        logger.info(f"Recall     : {results['recall']:.4f}")
        logger.info(f"F1 Score   : {results['f1']:.4f}")
        logger.info(f"ROC-AUC    : {results['roc_auc']:.4f}")
        logger.info(f"PR-AUC     : {results['pr_auc']:.4f}")

        logger.info("\nProject is Production Ready.")

    except Exception as e:

        logger.exception("Training Pipeline Failed.")

        raise e


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()
