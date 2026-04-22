# backend/utils/language_detector.py
# ============================================================
# Lightweight wrapper around langdetect
# Used by all scrapers before storing content
# ============================================================

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Makes langdetect deterministic (it's probabilistic by default)
DetectorFactory.seed = 42


def detect_language(text: str) -> str:
    """
    Detect language of given text.

    Args:
        text: Raw content string

    Returns:
        ISO 639-1 language code (e.g., 'en', 'fr', 'de')
        Returns 'unknown' if detection fails or text is too short
    """
    if not text or len(text.strip()) < 20:
        return "unknown"

    try:
        return detect(text[:2000])  # First 2000 chars is enough
    except LangDetectException:
        return "unknown"