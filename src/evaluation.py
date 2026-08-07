"""
===========================================================
Evaluation Module
===========================================================

This module is responsible for:

1. Model Evaluation
2. Classification Metrics
3. Confusion Matrix
4. ROC Curve
5. Precision-Recall Curve
6. Learning Curve
7. Validation Curve
8. Error Analysis
9. Benchmark Report

Author : Mahmoud Morsy
Project: Fake News Detection
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from logger import get_logger
from config import Config

logger = get_logger(__name__)


# ===========================================================
# Evaluation Dataclass
# ===========================================================

@dataclass
class EvaluationResult:

    accuracy: float

    precision: float

    recall: float

    f1: float

    roc_auc: float

    pr_auc: float

    mcc: float

    cohen_kappa: float

    predictions: np.ndarray

    decision_scores: np.ndarray


# ===========================================================
# Evaluate Model
# ===========================================================

def evaluate_model(
    model,
    X_test,
    y_test
) -> EvaluationResult:
    """
    Evaluate a trained model.
    """

    logger.info("="*60)
    logger.info("Evaluating Model")
    logger.info("="*60)

    y_pred = model.predict(X_test)

    scores = model.decision_function(X_test)

    return EvaluationResult(

        accuracy=accuracy_score(
            y_test,
            y_pred
        ),

        precision=precision_score(
            y_test,
            y_pred
        ),

        recall=recall_score(
            y_test,
            y_pred
        ),

        f1=f1_score(
            y_test,
            y_pred
        ),

        roc_auc=roc_auc_score(
            y_test,
            scores
        ),

        pr_auc=average_precision_score(
            y_test,
            scores
        ),

        mcc=matthews_corrcoef(
            y_test,
            y_pred
        ),

        cohen_kappa=cohen_kappa_score(
            y_test,
            y_pred
        ),

        predictions=y_pred,

        decision_scores=scores

    )


# ===========================================================
# Metrics Table
# ===========================================================

def metrics_table(
    result: EvaluationResult
) -> pd.DataFrame:

    return pd.DataFrame({

        "Metric":[

            "Accuracy",

            "Precision",

            "Recall",

            "F1",

            "ROC-AUC",

            "PR-AUC",

            "MCC",

            "Cohen Kappa"

        ],

        "Value":[

            result.accuracy,

            result.precision,

            result.recall,

            result.f1,

            result.roc_auc,

            result.pr_auc,

            result.mcc,

            result.cohen_kappa

        ]

    })


# ===========================================================
# Classification Report
# ===========================================================

def print_classification_report(
    y_test,
    result
):

    report = classification_report(

        y_test,

        result.predictions,

        digits=4

    )

    logger.info("\n"+report)

    return report


# ===========================================================
# Confusion Matrix
# ===========================================================

def plot_confusion_matrix(
    y_test,
    result,
    save=False
):

    cm = confusion_matrix(

        y_test,

        result.predictions

    )

    plt.figure(figsize=(6,5))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("True")

    plt.colorbar()

    if save:

        plt.savefig(

            Config.REPORT_DIR /

            "confusion_matrix.png",

            dpi=300,

            bbox_inches="tight"

        )

    plt.show()


# ===========================================================
# ROC Curve
# ===========================================================

def plot_roc_curve(
    y_test,
    result,
    save=False
):

    fpr,tpr,_ = roc_curve(

        y_test,

        result.decision_scores

    )

    plt.figure(figsize=(6,5))

    plt.plot(

        fpr,

        tpr,

        label=f"AUC={result.roc_auc:.4f}"

    )

    plt.plot(

        [0,1],

        [0,1],

        "--"

    )

    plt.legend()

    plt.grid(True)

    plt.title("ROC Curve")

    if save:

        plt.savefig(

            Config.REPORT_DIR/

            "roc_curve.png",

            dpi=300,

            bbox_inches="tight"

        )

    plt.show()


# ===========================================================
# Precision Recall Curve
# ===========================================================

def plot_pr_curve(
    y_test,
    result,
    save=False
):

    precision,recall,_ = precision_recall_curve(

        y_test,

        result.decision_scores

    )

    plt.figure(figsize=(6,5))

    plt.plot(

        recall,

        precision

    )

    plt.title("Precision Recall Curve")

    plt.grid(True)

    if save:

        plt.savefig(

            Config.REPORT_DIR/

            "pr_curve.png",

            dpi=300,

            bbox_inches="tight"

        )

    plt.show()


# ===========================================================
# Save Metrics
# ===========================================================

def save_metrics(
    result
):

    metrics_table(result).to_csv(

        Config.REPORT_DIR/

        "metrics.csv",

        index=False

    )

    logger.info("Metrics saved.")

# ===========================================================
# Learning Curve
# ===========================================================

from sklearn.model_selection import (
    learning_curve,
    validation_curve
)

from sklearn.pipeline import Pipeline


def plot_learning_curve(
    pipeline: Pipeline,
    X,
    y,
    cv: int = 5,
    scoring: str = "f1",
    save: bool = False
):
    """
    Plot Learning Curve.

    This function evaluates how model performance changes
    as the amount of training data increases.
    """

    logger.info("Generating Learning Curve...")

    train_sizes, train_scores, validation_scores = learning_curve(

        estimator=pipeline,

        X=X,

        y=y,

        cv=cv,

        scoring=scoring,

        train_sizes=np.linspace(0.1, 1.0, 10),

        n_jobs=-1

    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    plt.figure(figsize=(8,6))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training"
    )

    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation"
    )

    plt.xlabel("Training Examples")
    plt.ylabel("F1 Score")

    plt.title("Learning Curve")

    plt.grid(True)

    plt.legend()

    if save:

        plt.savefig(

            Config.REPORT_DIR /

            "learning_curve.png",

            dpi=300,

            bbox_inches="tight"

        )

    plt.show()


# ===========================================================
# Validation Curve
# ===========================================================

def plot_validation_curve(
    pipeline: Pipeline,
    X,
    y,
    param_name="classifier__C",
    param_range=None,
    cv=5,
    scoring="f1",
    save=False
):
    """
    Plot Validation Curve.

    Shows the effect of changing one hyperparameter.
    """

    logger.info("Generating Validation Curve...")

    if param_range is None:

        param_range = [

            0.01,

            0.1,

            1,

            10,

            100

        ]

    train_scores, validation_scores = validation_curve(

        estimator=pipeline,

        X=X,

        y=y,

        param_name=param_name,

        param_range=param_range,

        cv=cv,

        scoring=scoring,

        n_jobs=-1

    )

    train_mean = train_scores.mean(axis=1)
    validation_mean = validation_scores.mean(axis=1)

    plt.figure(figsize=(8,6))

    plt.plot(
        param_range,
        train_mean,
        marker="o",
        label="Training"
    )

    plt.plot(
        param_range,
        validation_mean,
        marker="o",
        label="Validation"
    )

    plt.xscale("log")

    plt.xlabel("C")

    plt.ylabel("F1 Score")

    plt.title("Validation Curve")

    plt.grid(True)

    plt.legend()

    if save:

        plt.savefig(

            Config.REPORT_DIR /

            "validation_curve.png",

            dpi=300,

            bbox_inches="tight"

        )

    plt.show()


# ===========================================================
# Confidence Analysis
# ===========================================================

def confidence_dataframe(
    result: EvaluationResult
) -> pd.DataFrame:
    """
    Build confidence dataframe using
    decision function scores.
    """

    confidence = np.abs(
        result.decision_scores
    )

    return pd.DataFrame({

        "Prediction": result.predictions,

        "Decision Score": result.decision_scores,

        "Confidence": confidence

    })


# ===========================================================
# Confidence Histogram
# ===========================================================

def plot_confidence_histogram(
    result,
    save=False
):
    """
    Plot confidence distribution.
    """

    df = confidence_dataframe(result)

    plt.figure(figsize=(8,5))

    plt.hist(

        df["Confidence"],

        bins=30

    )

    plt.title("Prediction Confidence")

    plt.xlabel("Confidence")

    plt.ylabel("Count")

    if save:

        plt.savefig(

            Config.REPORT_DIR/

            "confidence_histogram.png",

            dpi=300,

            bbox_inches="tight"

        )

    plt.show()


# ===========================================================
# Error Analysis
# ===========================================================

def error_dataframe(
    X_test,
    y_test,
    result
):
    """
    Create dataframe containing
    all predictions.
    """

    df = pd.DataFrame({

        "Text": X_test,

        "True Label": y_test,

        "Prediction": result.predictions,

        "Score": result.decision_scores

    })

    df["Correct"] = (

        df["True Label"]

        ==

        df["Prediction"]

    )

    return df


# ===========================================================
# Wrong Predictions
# ===========================================================

def wrong_predictions(
    X_test,
    y_test,
    result
):
    """
    Return all misclassified samples.
    """

    df = error_dataframe(

        X_test,

        y_test,

        result

    )

    return df[
        df["Correct"] == False
    ]


# ===========================================================
# Most Confident Errors
# ===========================================================

def most_confident_errors(
    X_test,
    y_test,
    result,
    top_n=10
):
    """
    Return the highest-confidence mistakes.
    """

    errors = wrong_predictions(

        X_test,

        y_test,

        result

    )

    errors["Confidence"] = np.abs(

        errors["Score"]

    )

    return errors.sort_values(

        "Confidence",

        ascending=False

    ).head(top_n)


# ===========================================================
# Least Confident Predictions
# ===========================================================

def least_confident_predictions(
    X_test,
    y_test,
    result,
    top_n=10
):
    """
    Return predictions closest
    to the decision boundary.
    """

    df = error_dataframe(

        X_test,

        y_test,

        result

    )

    df["Confidence"] = np.abs(

        df["Score"]

    )

    return df.sort_values(

        "Confidence"

    ).head(top_n)


# ===========================================================
# Save Evaluation Outputs
# ===========================================================

def save_reports(
    result
):
    """
    Save every evaluation report.
    """

    metrics = metrics_table(result)

    metrics.to_csv(

        Config.REPORT_DIR /

        "metrics.csv",

        index=False

    )

    logger.info(
        "Evaluation reports saved."
    )