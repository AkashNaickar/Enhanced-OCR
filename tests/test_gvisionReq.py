"""Tests for the Vision OCR batch pipeline using an injected fake client.

No Google Cloud credentials or network access are used; the ``google.cloud``
import is bypassed by passing a stub ``vision_module`` and ``client``.
"""

from __future__ import annotations

import csv
import types

import gvisionReq


class _Annotation:
    def __init__(self, description):
        self.description = description


class _Error:
    def __init__(self, message=""):
        self.message = message


class _Response:
    def __init__(self, text=None, error=""):
        self.error = _Error(error)
        self.text_annotations = [] if text is None else [_Annotation(text)]


class _FakeClient:
    """Returns queued responses in order and records which calls were made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self):
        if not self._responses:
            raise AssertionError("fake client ran out of queued responses")
        return self._responses.pop(0)

    def document_text_detection(self, image, image_context):
        self.calls.append("document")
        return self._next()

    def text_detection(self, image, image_context):
        self.calls.append("text")
        return self._next()


def _fake_vision_module():
    return types.SimpleNamespace(
        Image=lambda **kwargs: ("image", kwargs),
        ImageContext=lambda **kwargs: ("context", kwargs),
    )


def _write_image(folder, name):
    (folder / name).write_bytes(b"not-a-real-image")


def test_document_detection_output_is_cleaned(tmp_path):
    _write_image(tmp_path, "scan.jpg")
    client = _FakeClient([_Response(text="  Line one\n\nLine two  ")])
    out = tmp_path / "out.tsv"

    results = gvisionReq.extract_text_from_images(
        str(tmp_path),
        str(out),
        use_preprocessing=False,
        client=client,
        vision_module=_fake_vision_module(),
    )

    assert client.calls == ["document"]
    assert results == [{"filename": "scan.jpg", "text": "Line one\nLine two"}]

    with open(out, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows == [{"filename": "scan.jpg", "text": "Line one\nLine two"}]


def test_falls_back_to_text_detection_when_no_annotations(tmp_path):
    _write_image(tmp_path, "scan.png")
    client = _FakeClient([_Response(), _Response(text="Fallback text")])
    out = tmp_path / "out.tsv"

    results = gvisionReq.extract_text_from_images(
        str(tmp_path),
        str(out),
        use_preprocessing=False,
        client=client,
        vision_module=_fake_vision_module(),
    )

    assert client.calls == ["document", "text"]
    assert results[0]["text"] == "Fallback text"


def test_document_error_falls_back_to_text_detection(tmp_path):
    _write_image(tmp_path, "scan.png")
    client = _FakeClient([_Response(error="document failed"), _Response(text="Recovered")])

    results = gvisionReq.extract_text_from_images(
        str(tmp_path),
        str(tmp_path / "out.tsv"),
        use_preprocessing=False,
        client=client,
        vision_module=_fake_vision_module(),
    )

    assert client.calls == ["document", "text"]
    assert results[0]["text"] == "Recovered"


def test_response_error_yields_empty_text(tmp_path):
    _write_image(tmp_path, "scan.png")
    client = _FakeClient([_Response(error="document failed"), _Response(error="also failed")])

    results = gvisionReq.extract_text_from_images(
        str(tmp_path),
        str(tmp_path / "out.tsv"),
        use_preprocessing=False,
        client=client,
        vision_module=_fake_vision_module(),
    )

    assert results[0]["text"] == ""


def test_csv_output_uses_commas_for_non_tsv_extension(tmp_path):
    _write_image(tmp_path, "scan.jpg")
    client = _FakeClient([_Response(text="hello, world")])
    out = tmp_path / "out.csv"

    gvisionReq.extract_text_from_images(
        str(tmp_path),
        str(out),
        use_preprocessing=False,
        client=client,
        vision_module=_fake_vision_module(),
    )

    with open(out, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"filename": "scan.jpg", "text": "hello, world"}]
