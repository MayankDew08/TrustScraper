"""Text chunking helpers.

Primary strategy: keep paragraph boundaries when possible,
then fall back to word-window splitting for very long blocks.
"""

from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


def chunk_by_paragraphs(text: str, max_words: int = None) -> List[str]:
    """
    Split text into chunks by paragraph boundaries.
    Falls back to word-count splitting if paragraphs are too long.

    Args:
        text:      Full article text
        max_words: Max words per chunk (defaults to config value)

    Returns:
        List of text chunks, each under max_words
    """
    if not text or not text.strip():
        return []

    max_words = max_words or settings.chunk_size

    # Start by splitting on paragraph-like line breaks.
    raw_paragraphs = [
        p.strip()
        for p in text.split("\n")
            if p.strip() and len(p.strip()) > 30  # Drop tiny/noisy lines
    ]

    if not raw_paragraphs:
        return [text.strip()]

    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for paragraph in raw_paragraphs:
        para_words = paragraph.split()
        para_word_count = len(para_words)

        # If one paragraph is too large, slice it by word count.
        if para_word_count > max_words:
            # Flush any buffered text before slicing this paragraph.
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
                current_word_count = 0

            # Break the long paragraph into fixed-size sub-chunks.
            for i in range(0, para_word_count, max_words):
                sub_chunk = " ".join(para_words[i: i + max_words])
                chunks.append(sub_chunk)

        # If this paragraph would overflow, start a new chunk.
        elif current_word_count + para_word_count > max_words:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            current_chunk_words = para_words
            current_word_count = para_word_count

        # Otherwise keep accumulating text.
        else:
            current_chunk_words.extend(para_words)
            current_word_count += para_word_count

    # Flush remaining buffered content.
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return [c for c in chunks if c.strip()]


def chunk_text(text: str) -> List[str]:
    """
    Public interface for chunking.
    Handles edge cases before delegating to paragraph chunker.

    Edge cases handled:
    - Empty text -> returns []
    - Very short text -> returns one chunk
    - No clear paragraph breaks -> word-count splitting
    """
    if not text or not text.strip():
        return []

    # Short content can be returned as-is.
    if len(text.split()) <= settings.chunk_size:
        return [text.strip()]

    return chunk_by_paragraphs(text)