"""Tests for the T5 summarization helpers and import safety."""

from __future__ import annotations

import contextlib

import pytest

import t5transformer


class _FakeTokenizer:
    """Minimal stand-in that records the texts it is asked to tokenize."""

    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, texts, max_length=None, truncation=None, padding=None):
        self.calls.append(list(texts))
        return {"input_ids": [[1, 2] for _ in texts]}

    @contextlib.contextmanager
    def as_target_tokenizer(self):
        yield self


class _DecodingTokenizer:
    """Returns queued decodes, one per ``batch_decode`` call."""

    pad_token_id = 0

    def __init__(self, decodes):
        self._decodes = list(decodes)

    def batch_decode(self, values, skip_special_tokens=True):
        return self._decodes.pop(0)


def test_preprocess_function_prefixes_articles_and_labels_summaries():
    tokenizer = _FakeTokenizer()
    preprocess = t5transformer.build_preprocess_function(tokenizer, max_length=8)

    outputs = preprocess({"Articles": ["A", "B"], "Summaries": ["S1", "S2"]})

    assert tokenizer.calls[0] == ["summarize: A", "summarize: B"]
    assert tokenizer.calls[1] == ["S1", "S2"]
    assert outputs["labels"] == [[1, 2], [1, 2]]


def test_describe_lengths_prints_summary(capsys):
    t5transformer.describe_lengths("article", ["word " * 3])
    printed = capsys.readouterr().out
    assert "Longest article length: 3 words" in printed
    assert "Average article length: 3.00 words" in printed


def test_compute_metrics_returns_rouge_and_generation_length():
    np = pytest.importorskip("numpy")

    class _Rouge:
        def compute(self, predictions, references, use_stemmer, rouge_types):
            assert predictions == ["hello"]
            assert references == ["world"]
            assert use_stemmer is True
            assert list(rouge_types) == ["rouge1", "rouge2", "rougeL"]
            return {"rouge1": 0.5, "rouge2": 0.25, "rougeL": 0.4}

    tokenizer = _DecodingTokenizer([["hello"], ["world"]])
    metric = t5transformer.build_compute_metrics(tokenizer, _Rouge())

    eval_pred = type(
        "EvalPred",
        (),
        {
            "predictions": [np.array([[1, 2]])],
            "label_ids": np.array([[1, 2]]),
        },
    )()

    result = metric(eval_pred)

    assert result["rouge1"] == 0.5
    assert result["gen_len"] == 2.0
