"""
Text cleaning utilities for NLP preprocessing.
"""

from __future__ import annotations

import re
import unicodedata


HTML_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
MENTION_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def validate_text(text: str | None) -> str:
    """
    Validate the input text.
    """

    if not isinstance(text, str):
        return ""

    return text


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters.
    """

    return unicodedata.normalize("NFKC", text)


def to_lowercase(text: str) -> str:
    """
    Convert text to lowercase.
    """

    return text.lower()


def remove_html(text: str) -> str:
    """
    Remove HTML tags.
    """

    return HTML_PATTERN.sub("", text)


def remove_urls(text: str) -> str:
    """
    Remove URLs.
    """

    return URL_PATTERN.sub("", text)


def remove_emails(text: str) -> str:
    """
    Remove email addresses.
    """

    return EMAIL_PATTERN.sub("", text)


def remove_mentions(text: str) -> str:
    """
    Remove user mentions.
    """

    return MENTION_PATTERN.sub("", text)


def remove_hashtags(text: str) -> str:
    """
    Remove hashtags.
    """

    return HASHTAG_PATTERN.sub("", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize consecutive whitespace.
    """

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str | None) -> str:
    """
    Apply the complete text cleaning pipeline.
    """

    text = validate_text(text)
    text = normalize_unicode(text)
    text = to_lowercase(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = normalize_whitespace(text)

    return text