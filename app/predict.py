"""
===========================================================
Prediction Module
===========================================================

Responsible for:

1. Loading production artifacts
2. Cleaning input text
3. Making predictions
4. Returning confidence score
5. Returning human-readable labels

Author : Mahmoud Morsy
Project : Fake News Detection
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================

import numpy as np

from src.artifacts import (

    load_artifact,

    load_label_mapping

)

from src.preprocessing import clean_text

from src.config import Config

from src.logger import get_logger


logger = get_logger(__name__)


# ===========================================================
# Load Pipeline Once
# ===========================================================

logger.info("Loading production pipeline...")

pipeline = load_artifact(

    Config.PIPELINE_FILE

)

label_mapping = load_label_mapping()

logger.info("Pipeline loaded successfully.")


# ===========================================================
# Predict Single Text
# ===========================================================

def predict(text: str):
    """
    Predict a single news article.

    Returns
    -------
    dict
    """

    cleaned = clean_text(text)

    prediction = pipeline.predict([cleaned])[0]

    score = pipeline.decision_function([cleaned])[0]

    confidence = abs(score)

    return {

        "prediction": int(prediction),

        "label": label_mapping[str(prediction)],

        "decision_score": float(score),

        "confidence": float(confidence)

    }


# ===========================================================
# Predict Multiple Texts
# ===========================================================

def predict_batch(texts):
    """
    Predict multiple documents.
    """

    cleaned = [

        clean_text(text)

        for text in texts

    ]

    predictions = pipeline.predict(

        cleaned

    )

    scores = pipeline.decision_function(

        cleaned

    )

    results = []

    for pred, score in zip(

        predictions,

        scores

    ):

        results.append({

            "prediction": int(pred),

            "label": label_mapping[str(pred)],

            "decision_score": float(score),

            "confidence": float(abs(score))

        })

    return results


# ===========================================================
# Pretty Print
# ===========================================================

def print_prediction(result):

    print("=" * 50)

    print(f"Prediction      : {result['label']}")

    print(f"Confidence      : {result['confidence']:.4f}")

    print(f"Decision Score  : {result['decision_score']:.4f}")

    print("=" * 50)


# ===========================================================
# Example
# ===========================================================

if __name__ == "__main__":

    sample = """

    Reuters -
    The president announced a new economic policy today.

    """

    result = predict(sample)

    print_prediction(result)