"""Enhanced-OCR: Cloud Vision OCR plus T5-based article summarization.

The package only contains dependency-free helpers so it can be imported and
tested without Google Cloud credentials or the heavy ML runtime. The CLI entry
points live in ``gvisionReq.py`` and ``t5transformer.py`` at the repo root.
"""

from enhanced_ocr.textutils import (
    DEFAULT_SERVICE_ACCOUNT_FILE,
    SUPPORTED_IMAGE_EXTENSIONS,
    average_length,
    clean_extracted_text,
    find_longest_length,
    is_supported_image,
    list_images,
    make_summary_prompt,
    resolve_service_account_path,
    word_count,
)

__all__ = [
    "DEFAULT_SERVICE_ACCOUNT_FILE",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "average_length",
    "clean_extracted_text",
    "find_longest_length",
    "is_supported_image",
    "list_images",
    "make_summary_prompt",
    "resolve_service_account_path",
    "word_count",
]
