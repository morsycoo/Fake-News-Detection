"""
===========================================================
Unit Tests - Preprocessing
===========================================================
"""

import pandas as pd

from src.preprocessing import (
    clean_text,
    preprocess_dataset
)


def test_clean_text_lowercase():

    text = "HELLO WORLD"

    cleaned = clean_text(text)

    assert cleaned == "hello world"


def test_remove_url():

    text = "Visit https://example.com"

    cleaned = clean_text(text)

    assert "http" not in cleaned


def test_remove_numbers():

    text = "AI 2025"

    cleaned = clean_text(text)

    assert "2025" not in cleaned


def test_remove_punctuation():

    text = "Hello!!!"

    cleaned = clean_text(text)

    assert cleaned == "hello"


def test_preprocess_dataframe():

    df = pd.DataFrame({

        "text": [

            "Hello World!",

            "Fake News 2025"

        ],

        "label": [1,0]

    })

    processed = preprocess_dataset(df)

    assert len(processed) == 2

    assert processed["text"].isna().sum() == 0