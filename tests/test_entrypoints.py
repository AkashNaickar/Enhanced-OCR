"""Guard against importing the entry points requiring GCP or ML dependencies."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["enhanced_ocr", "enhanced_ocr.textutils", "gvisionReq", "t5transformer"],
)
def test_modules_import_without_optional_dependencies(module_name):
    assert importlib.import_module(module_name) is not None


def test_cli_entry_points_are_callable():
    gvision = importlib.import_module("gvisionReq")
    t5 = importlib.import_module("t5transformer")
    assert callable(gvision.main)
    assert callable(t5.main)
