"""
===========================================================
Unit Tests - Feature Engineering
===========================================================
"""

from src.feature_engineering import (

    build_vectorizer,

    build_selector

)


def test_vectorizer():

    vectorizer = build_vectorizer()

    assert vectorizer is not None


def test_selector():

    selector = build_selector()

    assert selector.k == 10000