# backend/utils/tagging.py
# ============================================================
# Topic tagging using KeyBERT (semantic) with RAKE fallback
# KeyBERT uses sentence-transformers to find keywords
# that are semantically close to the document meaning
# ============================================================

from typing import List
from keybert import KeyBERT
from rake_nltk import Rake
import nltk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Download required NLTK data silently
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Initialize models once at module level
# Why: Loading BERT models is expensive (~2-3 seconds)
# Loading once and reusing is standard production practice
_keybert_model = None


def _get_keybert() -> KeyBERT:
    """Lazy load KeyBERT model — only when first needed."""
    global _keybert_model
    if _keybert_model is None:
        # all-MiniLM-L6-v2: fast, small, good quality
        # Alternative: paraphrase-multilingual for non-English
        _keybert_model = KeyBERT(model="all-MiniLM-L6-v2")
    return _keybert_model


def extract_tags_keybert(text: str, max_tags: int = None) -> List[str]:
    """
    Extract semantic topic tags using KeyBERT.

    Args:
        text:     Article content
        max_tags: Maximum number of tags to return

    Returns:
        List of keyword strings
    """
    max_tags = max_tags or settings.max_tags

    if not text or len(text.strip()) < 50:
        return []

    try:
        kw_model = _get_keybert()

        # Extract keyphrases
        # diversity=0.5 reduces redundant similar tags
        keywords = kw_model.extract_keywords(
            text[:5000],  # Cap input for performance
            keyphrase_ngram_range=(
                settings.keyphrase_ngram_min,
                settings.keyphrase_ngram_max,
            ),
            stop_words="english",
            use_maxsum=True,
            nr_candidates=20,
            top_n=max_tags,
            diversity=0.5,
        )

        # keywords is list of (phrase, score) tuples
        return [kw[0].title() for kw in keywords]

    except Exception as e:
        print(f"[WARN] KeyBERT failed: {e}. Falling back to RAKE.")
        return extract_tags_rake(text, max_tags)


def extract_tags_rake(text: str, max_tags: int = None) -> List[str]:
    """
    Fallback keyword extractor using RAKE algorithm.
    RAKE = Rapid Automatic Keyword Extraction
    Frequency-based, no model needed.
    """
    max_tags = max_tags or settings.max_tags

    if not text or len(text.strip()) < 50:
        return []

    try:
        rake = Rake()
        rake.extract_keywords_from_text(text[:5000])
        phrases = rake.get_ranked_phrases()

        # Filter: max 3 words per tag, min 3 chars
        filtered = [
            p.title()
            for p in phrases
            if len(p.split()) <= 3 and len(p) >= 3
        ]

        return filtered[:max_tags]

    except Exception:
        return []


def extract_tags(text: str, max_tags: int = None) -> List[str]:
    """
    Public interface for tag extraction.
    Tries KeyBERT first, falls back to RAKE.
    """
    tags = extract_tags_keybert(text, max_tags)

    if not tags:
        tags = extract_tags_rake(text, max_tags)

    return tags