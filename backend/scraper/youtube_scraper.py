# backend/scraper/youtube_scraper.py
# ============================================================
# YouTube Scraper — Data API v3 + Supadata Transcript API
#
# Pipeline:
#   1. Extract video ID from URL
#   2. YouTube Data API → video metadata + channel info
#   3. Supadata API → transcript (primary)
#   4. youtube-transcript-api → transcript (fallback)
#   5. Description analysis → trust signals
#   6. NLP pipeline → language, tags, chunks
#
# Trust Score Signals:
#   author_credibility → subscribers + like/view ratio + credentials
#   citation_count     → views + likes (combined log-scaled)
#   transcript_source  → "supadata" | "official" | "none"
#                        none = penalized in trust score
# ============================================================

import json
import os
import sys
import re
import time
import math
from datetime import datetime, timezone
from typing import Optional, Tuple
from urllib.parse import urlparse, parse_qs

from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from supadata import Supadata, SupadataError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.chunking import chunk_text
from utils.tagging import extract_tags
from utils.language_detector import detect_language


# ============================================================
# API Clients
# ============================================================

def _get_youtube_client():
    """Build YouTube Data API v3 client."""
    if not settings.youtube_api_key:
        raise ValueError(
            "YOUTUBE_API_KEY not set. Add it to your .env file."
        )
    return build(
        "youtube", "v3",
        developerKey=settings.youtube_api_key,
    )


def _get_supadata_client() -> Optional[Supadata]:
    """
    Build Supadata client.
    Returns None if API key not configured.
    """
    supadata_api_key = getattr(settings, "supadata_api_key", "")
    if not supadata_api_key:
        print(f"[SUPADATA] ⚠ SUPADATA_API_KEY not set in .env")
        return None
    return Supadata(api_key=supadata_api_key)

# ============================================================
# Video ID Extraction
# ============================================================

def _extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from any YouTube URL format.

    Handles:
      youtube.com/watch?v=ID
      youtu.be/ID
      youtube.com/embed/ID
      youtube.com/shorts/ID
      Raw 11-char video ID
    """
    if not url:
        return None

    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            params = parse_qs(parsed.query)
            return params.get("v", [None])[0]
        for prefix in ("/embed/", "/v/", "/shorts/"):
            if parsed.path.startswith(prefix):
                return (
                    parsed.path
                    .split(prefix)[1]
                    .split("/")[0]
                    .split("?")[0]
                )

    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("?")[0]

    # Raw 11-char video ID
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url

    return None


# ============================================================
# Metadata Fetching
# ============================================================

def _fetch_video_metadata(youtube, video_id: str) -> dict:
    """
    Fetch video metadata from YouTube Data API v3.

    Returns:
        dict with title, channel, date, views, likes,
        comments, description, source_url
    """
    print(f"[YT-API] Fetching video: {video_id}")

    try:
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id,
        ).execute()

        items = response.get("items", [])
        if not items:
            print(f"[YT-API] ❌ Video not found: {video_id}")
            return {}

        video   = items[0]
        snippet = video.get("snippet", {})
        stats   = video.get("statistics", {})

        # Parse publish date
        pub_raw = snippet.get("publishedAt", "")
        try:
            pub_date = datetime.fromisoformat(
                pub_raw.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        except Exception:
            pub_date = "Unknown"

        result = {
            "title":          snippet.get("title", "Unknown Title"),
            "channel_name":   snippet.get("channelTitle", "Unknown Channel"),
            "channel_id":     snippet.get("channelId", ""),
            "published_date": pub_date,
            "description":    snippet.get("description", ""),
            "view_count":     int(stats.get("viewCount",    0)),
            "like_count":     int(stats.get("likeCount",    0)),
            "comment_count":  int(stats.get("commentCount", 0)),
            "video_id":       video_id,
            "source_url":     f"https://www.youtube.com/watch?v={video_id}",
        }

        print(f"[YT-API] ✅ Title   : {result['title'][:60]}")
        print(f"[YT-API] ✅ Channel : {result['channel_name']}")
        print(f"[YT-API] ✅ Date    : {result['published_date']}")
        print(f"[YT-API] ✅ Views   : {result['view_count']:,}")
        print(f"[YT-API] ✅ Likes   : {result['like_count']:,}")
        print(f"[YT-API] ✅ Comments: {result['comment_count']:,}")

        return result

    except Exception as e:
        print(f"[YT-API] ❌ Video metadata failed: {e}")
        return {}


def _fetch_channel_info(youtube, channel_id: str) -> dict:
    """
    Fetch channel metadata from YouTube Data API v3.

    Returns:
        dict with subscriber_count, total_videos,
        channel_description, subscribers_hidden
    """
    if not channel_id:
        return {
            "subscriber_count":   0,
            "channel_description":"",
            "total_videos":       0,
            "total_views":        0,
            "subscribers_hidden": False,
        }

    print(f"[YT-API] Fetching channel: {channel_id}")

    try:
        response = youtube.channels().list(
            part="snippet,statistics",
            id=channel_id,
        ).execute()

        items = response.get("items", [])
        if not items:
            return {
                "subscriber_count":   0,
                "channel_description":"",
                "total_videos":       0,
                "total_views":        0,
                "subscribers_hidden": False,
            }

        ch      = items[0]
        stats   = ch.get("statistics", {})
        snippet = ch.get("snippet", {})
        hidden  = stats.get("hiddenSubscriberCount", False)
        subs    = 0 if hidden else int(stats.get("subscriberCount", 0))

        result = {
            "subscriber_count":    subs,
            "channel_description": snippet.get("description", ""),
            "total_videos":        int(stats.get("videoCount", 0)),
            "total_views":         int(stats.get("viewCount",  0)),
            "subscribers_hidden":  hidden,
        }

        print(
            f"[YT-API] ✅ Subscribers: "
            f"{subs:,} {'(hidden)' if hidden else ''}"
        )
        print(f"[YT-API] ✅ Videos     : {result['total_videos']:,}")

        return result

    except Exception as e:
        print(f"[YT-API] ⚠ Channel info failed: {e}")
        return {
            "subscriber_count":   0,
            "channel_description":"",
            "total_videos":       0,
            "total_views":        0,
            "subscribers_hidden": False,
        }


# ============================================================
# Transcript Fetching
# ============================================================

def _fetch_transcript_supadata(url: str) -> Tuple[str, str]:
    """
    Fetch transcript using Supadata API.

    Supadata supports:
    - YouTube, TikTok, Instagram, X (Twitter)
    - Multiple languages
    - Native + auto-generated captions
    - Returns plain text when text=True

    Why Supadata as primary?
    → More reliable than youtube-transcript-api
       (handles new YouTube caption formats)
    → Simple clean API
    → Returns plain text directly
    → Supports multiple platforms

    Returns:
        (transcript_text: str, source: str)
        source = "supadata" | "failed"
    """
    client = _get_supadata_client()
    if not client:
        return "", "failed"

    print(f"[SUPADATA] Fetching transcript: {url}")

    try:
        transcript = client.transcript(
            url=url,
            lang="en",    # Prefer English
            text=True,    # Return plain text not timestamped chunks
            mode="auto",  # Try native first, fallback to auto-generated
        )

        # Handle immediate result
        if hasattr(transcript, "content") and transcript.content:
            text = transcript.content
            lang = getattr(transcript, "lang", "en")
            word_count = len(text.split())
            print(f"[SUPADATA] ✅ Language: {lang}")
            print(f"[SUPADATA] ✅ Words   : {word_count:,}")
            print(f"[SUPADATA] Preview   : {text[:200]}...")
            return text, "supadata"

        # Handle async job (large files)
        if hasattr(transcript, "job_id"):
            print(f"[SUPADATA] Async job: {transcript.job_id}")
            print(f"[SUPADATA] ⚠ Async not supported in this pipeline")
            return "", "failed"

        print(f"[SUPADATA] ⚠ Empty response")
        return "", "failed"

    except SupadataError as e:
        print(f"[SUPADATA] ⚠ API error: {e}")
        return "", "failed"
    except Exception as e:
        print(f"[SUPADATA] ⚠ Failed: {e}")
        return "", "failed"


def _fetch_transcript_official(video_id: str) -> Tuple[str, str]:
    """
    Fetch transcript using youtube-transcript-api.
    Fallback when Supadata fails.

    Tries:
    1. English (manual or auto)
    2. Any available language

    Returns:
        (transcript_text: str, source: str)
        source = "official" | "failed"
    """
    print(f"[YT-TRANSCRIPT] Trying official API: {video_id}")

    # Try English directly
    try:
        entries = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en", "en-US", "en-GB"],
        )
        text = " ".join(
            e.get("text", "") for e in entries
        ).strip()

        if text and len(text.split()) > 50:
            print(
                f"[YT-TRANSCRIPT] ✅ English: "
                f"{len(text.split()):,} words"
            )
            return text, "official"

    except TranscriptsDisabled:
        print(f"[YT-TRANSCRIPT] Transcripts disabled")
        return "", "failed"
    except Exception as e:
        print(f"[YT-TRANSCRIPT] English failed: {type(e).__name__}")

    # Try any available language
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available = [t.language_code for t in transcript_list]
        print(f"[YT-TRANSCRIPT] Available: {available}")

        for lang in available:
            try:
                entries = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=[lang]
                )
                text = " ".join(
                    e.get("text", "") for e in entries
                ).strip()

                if text and len(text.split()) > 50:
                    print(
                        f"[YT-TRANSCRIPT] ✅ {lang}: "
                        f"{len(text.split()):,} words"
                    )
                    return text, "official"
            except Exception:
                continue

    except VideoUnavailable:
        print(f"[YT-TRANSCRIPT] Video unavailable")
    except Exception as e:
        print(f"[YT-TRANSCRIPT] List failed: {e}")

    return "", "failed"


def _fetch_transcript(url: str, video_id: str) -> Tuple[str, str]:
    """
    Multi-strategy transcript fetcher.

    Priority:
      1. Supadata API (most reliable, primary)
      2. youtube-transcript-api (official fallback)
      3. No transcript → empty string

    transcript_source stored in metadata:
      "supadata"  → best quality, no penalty
      "official"  → direct from YouTube, no penalty
      "none"      → no transcript, content from description

    Returns:
        (transcript_text: str, transcript_source: str)
    """
    print(f"\n[TRANSCRIPT] Starting for: {video_id}")

    # ── Strategy 1: Supadata API ───────────────────────────
    text, source = _fetch_transcript_supadata(url)
    if text and len(text.split()) > 50:
        return _clean_transcript(text), source

    print(f"[TRANSCRIPT] Supadata failed → trying official API")

    # ── Strategy 2: youtube-transcript-api ────────────────
    text, source = _fetch_transcript_official(video_id)
    if text and len(text.split()) > 50:
        return _clean_transcript(text), source

    print(f"[TRANSCRIPT] ❌ No transcript available")
    return "", "none"


def _clean_transcript(text: str) -> str:
    """
    Clean raw transcript text.

    Removes:
    - [Music], [Applause], [Laughter] tags
    - Excessive whitespace
    - Repeated punctuation
    - HTML entities
    """
    if not text:
        return ""

    # Remove bracketed annotations
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)

    # Remove HTML entities
    text = (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
    )

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# Description Analysis
# ============================================================

def _analyze_description(description: str) -> dict:
    """
    Extract trust signals from video description.

    Checks:
    - Academic citation links (PubMed, DOI, journals)
    - Medical disclaimer text
    - Promotional spam indicators
    - Total URL count

    These signals feed directly into trust scoring.
    """
    desc_lower = description.lower()

    # Academic links in description = high credibility signal
    academic_patterns = [
        r'pubmed\.ncbi\.nlm\.nih\.gov',
        r'doi\.org/10\.',
        r'ncbi\.nlm\.nih\.gov',
        r'scholar\.google\.com',
        r'nature\.com/articles',
        r'thelancet\.com',
        r'nejm\.org',
        r'jamanetwork\.com',
        r'bmj\.com',
    ]
    citation_links = sum(
        len(re.findall(p, desc_lower))
        for p in academic_patterns
    )

    # Medical disclaimer in description
    disclaimer_phrases = [
        "not medical advice",
        "consult a doctor",
        "for educational purposes",
        "informational purposes only",
        "healthcare professional",
        "not a substitute",
        "always consult",
        "speak with a doctor",
        "not intended to diagnose",
    ]
    has_disclaimer = any(p in desc_lower for p in disclaimer_phrases)

    # Promotional spam signals
    spam_signals = [
        "use my code", "promo code", "discount",
        "affiliate", "sponsored", "ad ",
        "buy now", "limited time offer",
        "click the link below", "link in bio",
    ]
    spam_count = sum(1 for s in spam_signals if s in desc_lower)

    return {
        "citation_links":  citation_links,
        "has_disclaimer":  has_disclaimer,
        "spam_signals":    spam_count,
        "total_urls":      len(re.findall(r'https?://', description)),
    }


# ============================================================
# Helpers
# ============================================================

def _detect_medical_disclaimer(text: str) -> bool:
    """Check for medical disclaimer language."""
    phrases = [
        "not medical advice", "consult a doctor",
        "not a substitute", "healthcare professional",
        "for informational purposes", "always consult",
        "not intended to diagnose", "educational purposes only",
        "speak with a doctor", "seek professional",
    ]
    lower = text.lower()
    return any(p in lower for p in phrases)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_result(url: str, error_msg: str) -> dict:
    """Safe error dict — pipeline never crashes."""
    return {
        "source_url":     url,
        "source_type":    "youtube",
        "title":          "Unknown",
        "author":         "Unknown Channel",
        "published_date": "Unknown",
        "language":       "unknown",
        "region":         "Unknown",
        "topic_tags":     [],
        "trust_score":    None,
        "content_chunks": [],
        "metadata": {
            "domain":            "youtube.com",
            "has_transcript":    False,
            "transcript_source": "none",
            "error":             error_msg,
            "scraped_at":        _now_iso(),
        },
    }


# ============================================================
# Core Scraper
# ============================================================

def scrape_youtube(url: str) -> dict:
    """
    Scrape a single YouTube video.

    Full pipeline:
      1. Extract video ID from URL
      2. YouTube Data API → video metadata
      3. YouTube Data API → channel metadata
      4. Supadata API → transcript (primary)
         youtube-transcript-api → transcript (fallback)
      5. Description analysis → trust signals
      6. NLP pipeline → language, tags, chunks
      7. Return structured dict matching schema

    Trust scoring signals stored in metadata:
      - view_count, like_count → engagement scoring
      - subscriber_count       → author credibility
      - transcript_source      → reliability flag
      - description_analysis   → citation + disclaimer signals
      - has_transcript         → content completeness

    Args:
        url: YouTube video URL

    Returns:
        Dict matching assignment JSON schema
    """
    print(f"\n{'='*60}")
    print(f"[YOUTUBE] {url}")
    print(f"{'='*60}")

    # ── Step 1: Video ID ───────────────────────────────────
    video_id = _extract_video_id(url)
    if not video_id:
        print(f"[ERROR] Cannot extract video ID: {url}")
        return _error_result(url, "Invalid YouTube URL")

    print(f"[YOUTUBE] Video ID: {video_id}")

    # ── Step 2: YouTube API client ─────────────────────────
    try:
        youtube = _get_youtube_client()
    except Exception as e:
        return _error_result(url, f"YouTube API init failed: {e}")

    # ── Step 3: Video metadata ─────────────────────────────
    print(f"\n[STEP 1] Video metadata...")
    video_meta = _fetch_video_metadata(youtube, video_id)
    if not video_meta:
        return _error_result(url, "Video metadata fetch failed")

    # ── Step 4: Channel metadata ───────────────────────────
    print(f"\n[STEP 2] Channel metadata...")
    channel_id   = video_meta.get("channel_id", "")
    channel_info = _fetch_channel_info(youtube, channel_id)

    # ── Step 5: Transcript ─────────────────────────────────
    print(f"\n[STEP 3] Transcript...")
    transcript, transcript_source = _fetch_transcript(url, video_id)
    has_transcript = len(transcript.split()) > 50

    print(f"[TRANSCRIPT] Source : {transcript_source}")
    print(f"[TRANSCRIPT] Words  : {len(transcript.split()):,}")

    # ── Step 6: Description analysis ──────────────────────
    print(f"\n[STEP 4] Description analysis...")
    description   = video_meta.get("description", "")
    desc_analysis = _analyze_description(description)

    print(f"[DESC] Citation links : {desc_analysis['citation_links']}")
    print(f"[DESC] Has disclaimer : {desc_analysis['has_disclaimer']}")
    print(f"[DESC] Spam signals   : {desc_analysis['spam_signals']}")

    # ── Step 7: Choose content for NLP ────────────────────
    # Transcript > Description > Title
    if has_transcript:
        content = transcript
        print(f"\n[CONTENT] Transcript: {len(content.split()):,} words")
    elif description and len(description.split()) > 30:
        content = description
        print(f"\n[CONTENT] Description: {len(content.split()):,} words")
    else:
        content = video_meta.get("title", "")
        print(f"\n[CONTENT] Title only: {len(content.split()):,} words")

    word_count = len(content.split())

    # ── Step 8: NLP Pipeline ──────────────────────────────
    print(f"\n[NLP] Processing {word_count:,} words...")

    language = detect_language(content)
    print(f"[NLP] Language  : {language}")

    # Check both transcript + description for disclaimer
    full_text      = f"{transcript} {description}".strip()
    has_disclaimer = _detect_medical_disclaimer(full_text)
    print(f"[NLP] Disclaimer: {has_disclaimer}")

    print(f"[NLP] Extracting tags (KeyBERT)...")
    tags = extract_tags(content)
    print(f"[NLP] Tags      : {tags}")

    print(f"[NLP] Chunking...")
    chunks = chunk_text(content)
    print(f"[NLP] Chunks    : {len(chunks)}")

    # ── Step 9: Polite delay ───────────────────────────────
    time.sleep(1)

    return {
        "source_url":     video_meta["source_url"],
        "source_type":    "youtube",
        "title":          video_meta["title"],
        "author":         video_meta["channel_name"],
        "published_date": video_meta["published_date"],
        "language":       language,
        "region":         "Unknown",
        "topic_tags":     tags,
        "trust_score":    None,
        "content_chunks": chunks,
        "metadata": {
            "domain":                 "youtube.com",
            "video_id":               video_id,
            "channel_id":             channel_id,
            "has_medical_disclaimer": has_disclaimer,
            "has_transcript":         has_transcript,
            "transcript_source":      transcript_source,
            "word_count":             word_count,
            "chunk_count":            len(chunks),
            "view_count":             video_meta.get("view_count",   0),
            "like_count":             video_meta.get("like_count",   0),
            "comment_count":          video_meta.get("comment_count",0),
            "subscriber_count":       channel_info.get("subscriber_count",    0),
            "subscribers_hidden":     channel_info.get("subscribers_hidden", False),
            "channel_description":    channel_info.get("channel_description","")[:500],
            "channel_total_videos":   channel_info.get("total_videos",        0),
            "description_preview":    description[:300],
            "description_analysis":   desc_analysis,
            "scraped_at":             _now_iso(),
        },
    }


# ============================================================
# Public API
# ============================================================

def scrape_all_youtube(urls: list) -> list:
    """
    Scrape list of YouTube video URLs.
    Returns list of structured result dicts.
    """
    results = []
    total   = len(urls)

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{total}] Processing...")
        result = scrape_youtube(url)
        results.append(result)

    return results


# ============================================================
# Direct Runner
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  YouTube Scraper — Data API + Supadata Transcript")
    print("=" * 60)
    print("\n  Enter 2 YouTube video URLs:\n")

    urls = []
    for i in range(2):
        while True:
            raw = input(f"  URL {i+1}: ").strip()
            if "youtu" in raw or re.match(r'^[A-Za-z0-9_-]{11}$', raw):
                urls.append(raw)
                break
            print("  ⚠  Must be a valid YouTube URL")

    print(f"\n▶ Scraping {len(urls)} videos...\n")

    results = scrape_all_youtube(urls)

    # ── Save ───────────────────────────────────────────────
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "youtube.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary Table ──────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"  ✅ Saved → {output_path}")
    print("=" * 95)

    print(
        f"\n{'#':<3} {'Channel':<22} {'Title':<28} "
        f"{'Views':>9} {'Subs':>9} {'Words':>7} "
        f"{'Transcript':<12} {'Tags':>5} Status"
    )
    print("-" * 100)

    for i, r in enumerate(results, 1):
        channel  = r.get("author",  "?")[:21]
        title    = r.get("title",   "?")[:27]
        meta     = r.get("metadata", {})
        views    = meta.get("view_count",       0)
        subs     = meta.get("subscriber_count", 0)
        words    = meta.get("word_count",        0)
        src      = meta.get("transcript_source", "none")
        tags     = len(r.get("topic_tags", []))
        chunks   = meta.get("chunk_count",       0)
        ok       = "✅" if chunks > 0 else "❌"
        err      = meta.get("error", "")

        src_label = {
            "supadata": "✅ Supadata",
            "official": "✅ Official",
            "none":     "❌ None",
        }.get(src, f"⚠ {src}")

        print(
            f"{i:<3} {channel:<22} {title:<28} "
            f"{views:>9,} {subs:>9,} {words:>7,} "
            f"{src_label:<12} {tags:>5}  {ok}"
        )
        if err:
            print(f"    └─ Error: {err[:60]}")

    print()