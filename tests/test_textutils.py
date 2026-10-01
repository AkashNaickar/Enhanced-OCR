"""Tests for the dependency-free helpers in ``enhanced_ocr.textutils``."""

from __future__ import annotations

import pytest

from enhanced_ocr import textutils


def test_is_supported_image_is_case_insensitive():
    assert textutils.is_supported_image("scan.JPG")
    assert textutils.is_supported_image("page.jpeg")
    assert textutils.is_supported_image("photo.TIFF")
    assert not textutils.is_supported_image("notes.txt")
    assert not textutils.is_supported_image("archive.tar.gz")


def test_list_images_filters_and_sorts(tmp_path):
    for name in ["b.png", "a.JPEG", "notes.txt", "c.gif"]:
        (tmp_path / name).write_bytes(b"x")

    assert textutils.list_images(str(tmp_path)) == ["a.JPEG", "b.png", "c.gif"]


def test_list_images_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        textutils.list_images(str(tmp_path / "does-not-exist"))


def test_clean_extracted_text_strips_and_drops_blank_lines():
    raw = "  First line\r\n\r\n   \nSecond line  "
    assert textutils.clean_extracted_text(raw) == "First line\nSecond line"


@pytest.mark.parametrize("value", [None, "", "   \n\n  "])
def test_clean_extracted_text_empty_inputs(value):
    assert textutils.clean_extracted_text(value) == ""


def test_make_summary_prompt_adds_prefix():
    assert textutils.make_summary_prompt("Body text") == "summarize: Body text"


def test_word_count_ignores_extra_whitespace():
    assert textutils.word_count("one  two\tthree\nfour") == 4


def test_find_longest_length_counts_buckets():
    texts = ["word " * 4001, "word " * 10]
    longest, over_4k, over_2k, over_1k, over_500 = textutils.find_longest_length(texts)
    assert longest == 4001
    assert (over_4k, over_2k, over_1k, over_500) == (1, 1, 1, 1)


def test_average_length_handles_empty_and_values():
    assert textutils.average_length([]) == 0.0
    assert textutils.average_length(["a b", "c d e"]) == 2.5


def test_resolve_service_account_path_default(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("VISION_SERVICE_ACCOUNT_FILE", raising=False)
    assert textutils.resolve_service_account_path() == textutils.DEFAULT_SERVICE_ACCOUNT_FILE


def test_resolve_service_account_path_prefers_google_credentials(monkeypatch):
    monkeypatch.setenv("VISION_SERVICE_ACCOUNT_FILE", "fallback.json")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "primary.json")
    assert textutils.resolve_service_account_path() == "primary.json"


def test_resolve_service_account_path_accepts_explicit_mapping():
    assert (
        textutils.resolve_service_account_path({"VISION_SERVICE_ACCOUNT_FILE": "x.json"})
        == "x.json"
    )
