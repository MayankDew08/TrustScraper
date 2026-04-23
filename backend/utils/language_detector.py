# Small wrapper around langdetect.

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Make results reproducible; langdetect is probabilistic otherwise.
DetectorFactory.seed = 42


def detect_language(text: str) -> str:
    """
    Detect language of given text.

    Args:
        text: Raw content string

    Returns:
        ISO 639-1 language code (for example: 'en', 'fr', 'de')
        Returns 'unknown' if detection fails or text is too short
    """
    if not text or len(text.strip()) < 20:
        return "unknown"

    try:
        return detect(text[:2000])  # A short prefix is usually sufficient.
    except LangDetectException:
        return "unknown"