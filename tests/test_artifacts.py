"""
===========================================================
Unit Tests - Artifacts
===========================================================
"""

from pathlib import Path

from src.config import Config


def test_artifact_folder():

    assert Config.ARTIFACT_DIR.exists()


def test_reports_folder():

    assert Config.REPORT_DIR.exists()