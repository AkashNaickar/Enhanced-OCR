"""Pure text and path helpers shared by the OCR and summarization entry points.

Nothing in this module talks to Google Cloud, OpenCV, Pillow or PyTorch, so it
can be imported and unit tested without any runtime dependencies or
credentials.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence

SUPPORTED_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
)

DEFAULT_SERVICE_ACCOUNT_FILE = "vision_ocr.json"

# Checked in order; the first non-empty value wins.
CREDENTIAL_ENV_VARS = ("GOOGLE_APPLICATION_CREDENTIALS", "VISION_SERVICE_ACCOUNT_FILE")


def is_supported_image(filename: str) -> bool:
    """Return True if *filename* has an image extension Cloud Vision accepts."""
    return filename.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)


def list_images(folder: str) -> list[str]:
    """Return supported image filenames in *folder*, sorted for stable output."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(name for name in os.listdir(folder) if is_supported_image(name))


def clean_extracted_text(text: str | None) -> str:
    """Normalise Vision output: strip the blob and drop whitespace-only lines."""
    if not text:
        return ""
    normalised = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line for line in normalised.splitlines() if line.strip())


def make_summary_prompt(article: str) -> str:
    """Prefix an article the way the T5 model is trained to expect."""
    return f"summarize: {article}"


def word_count(text: str) -> int:
    """Number of whitespace-separated tokens in *text*."""
    return len(text.split())


def find_longest_length(texts: Iterable[str]) -> tuple[int, int, int, int, int]:
    """Return ``(max words, count>4000, count>2000, count>1000, count>500)``."""
    max_length = 0
    counter_4k = counter_2k = counter_1k = counter_500 = 0
    for text in texts:
        length = word_count(text)
        if length > 4000:
            counter_4k += 1
        if length > 2000:
            counter_2k += 1
        if length > 1000:
            counter_1k += 1
        if length > 500:
            counter_500 += 1
        if length > max_length:
            max_length = length
    return max_length, counter_4k, counter_2k, counter_1k, counter_500


def average_length(texts: Sequence[str]) -> float:
    """Mean word count of *texts*, or ``0.0`` for an empty input."""
    items = list(texts)
    if not items:
        return 0.0
    return sum(word_count(text) for text in items) / len(items)


def resolve_service_account_path(env: Mapping[str, str] | None = None) -> str:
    """Resolve the service-account key path from the environment.

    ``GOOGLE_APPLICATION_CREDENTIALS`` takes precedence over
    ``VISION_SERVICE_ACCOUNT_FILE``; when neither is set the historical
    ``vision_ocr.json`` filename is used.
    """
    source = os.environ if env is None else env
    for name in CREDENTIAL_ENV_VARS:
        value = source.get(name)
        if value:
            return value
    return DEFAULT_SERVICE_ACCOUNT_FILE
