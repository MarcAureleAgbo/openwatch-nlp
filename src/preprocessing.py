"""Text preprocessing utilities for OpenWatch NLP."""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Return a normalized text string suitable for TF-IDF vectorization."""
    if not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"http\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
