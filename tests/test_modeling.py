"""
===========================================================
Unit Tests - Modeling
===========================================================
"""

from src.modeling import build_model


def test_linear_svm():

    model = build_model("linear_svm")

    assert model.__class__.__name__ == "LinearSVC"


def test_logistic():

    model = build_model("logistic_regression")

    assert model.__class__.__name__ == "LogisticRegression"


def test_nb():

    model = build_model("naive_bayes")

    assert model.__class__.__name__ == "MultinomialNB"