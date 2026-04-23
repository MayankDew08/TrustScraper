# backend/scraper/blog_scraper.py
# ============================================================
# Multi-source blog scraper
#
# Supported sources:
#   1. Medium.com    → Playwright (full JS rendering)
#   2. Nature.com    → Springer API + requests/BS4
#
# Two-phase pipeline:
#   Phase 1: Scrape article content + metadata
#   Phase 2: Scrape author profiles
#
# Why two phases?
#   Medium: author about page needs separate browser session
#   Nature: author data comes from API (done in Phase 1)
# ============================================================

import json
import os
import sys
import re
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse, parse_qs

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Page,
)
from dateutil import parser as date_parser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.chunking import chunk_text
from utils.tagging import extract_tags
from utils.language_detector import detect_language


# ============================================================
# Springer API Constants
# ============================================================

SPRINGER_META_URL = "https://api.springernature.com/meta/v2/json"
SPRINGER_OA_URL   = "https://api.springernature.com/openaccess/json"


# ============================================================
# Shared Utilities
# ============================================================

def _get_domain(url: str) -> str:
    """Extract clean domain (no www.) from URL."""
    return urlparse(url).netloc.replace("www.", "").lower()


def _now_iso() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_count_text(text: str) -> int:
    """
    Convert count strings to integers.
    "2.1K" → 2100, "499" → 499, "1.5M" → 1500000
    """
    if not text:
        return 0
    text = text.strip().replace(",", "")
    match = re.search(r'([\d]+\.?[\d]*)\s*([KkMm]?)', text)
    if not match:
        return 0
    try:
        number = float(match.group(1))
        suffix = match.group(2).upper()
        if suffix == "K":
            number *= 1_000
        elif suffix == "M":
            number *= 1_000_000
        return int(number)
    except Exception:
        return 0


def _extract_username(article_url: str) -> str:
    """
    Extract Medium @username from article URL.
    "https://medium.com/@Dr.Shlain/article" → "Dr.Shlain"
    """
    try:
        match = re.search(r'medium\.com/@([^/]+)', article_url)
        if match:
            return match.group(1)
    except Exception:
        pass
    return ""


def _detect_medical_disclaimer(text: str) -> bool:
    """Detect medical disclaimer language in content."""
    phrases = [
        "consult a doctor", "consult your physician",
        "not a substitute", "healthcare professional",
        "seek medical attention", "talk to your doctor",
        "not medical advice", "for informational purposes only",
        "always consult", "speak with a qualified",
        "not intended to diagnose", "medical professional",
        "before starting any", "consult with a",
        "professional medical advice",
    ]
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def _clean_raw_text(raw_text: str) -> str:
    """
    Clean Playwright inner_text() output.
    Remove nav items, duplicates, low-content lines.
    """
    lines = raw_text.split("\n")
    seen    = set()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 30:
            continue
        if line in seen:
            continue
        if len(line) > 0:
            alpha_ratio = sum(c.isalpha() for c in line) / len(line)
            if alpha_ratio < 0.40:
                continue
        seen.add(line)
        cleaned.append(line)

    return "\n".join(cleaned)


def _error_result(url: str, error_msg: str) -> dict:
    """Safe error dict — pipeline never crashes."""
    return {
        "source_url":     url,
        "source_type":    "blog",
        "title":          "Unknown",
        "author":         "Unknown Author",
        "published_date": "Unknown",
        "language":       "unknown",
        "region":         "Unknown",
        "topic_tags":     [],
        "trust_score":    None,
        "content_chunks": [],
        "metadata": {
            "domain":                 _get_domain(url),
            "has_medical_disclaimer": False,
            "word_count":             0,
            "chunk_count":            0,
            "clap_count":             0,
            "author_profile": {
                "followers":     0,
                "bio":           "",
                "article_count": 0,
                "profile_url":   "",
            },
            "error":      error_msg,
            "scraped_at": _now_iso(),
        },
    }


# ============================================================
# Playwright Helpers
# ============================================================

def _navigate(page: Page, url: str):
    """Navigate with networkidle → domcontentloaded fallback."""
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
    except PWTimeout:
        print(f"[WARN] networkidle timeout → domcontentloaded")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
        except PWTimeout:
            print(f"[WARN] domcontentloaded timeout → using page as-is")
    page.wait_for_timeout(2000)


def _scroll_page(page: Page):
    """Scroll entire page to trigger lazy loading."""
    page.evaluate("""
        async () => {
            await new Promise((resolve) => {
                let totalHeight = 0;
                const distance = 300;
                const timer = setInterval(() => {
                    window.scrollBy(0, distance);
                    totalHeight += distance;
                    if (totalHeight >= document.body.scrollHeight) {
                        clearInterval(timer);
                        resolve();
                    }
                }, 100);
            });
        }
    """)
    page.wait_for_timeout(1000)


# ============================================================
# Medium — Metadata Extraction (BS4)
# ============================================================

def _extract_json_ld(soup: BeautifulSoup) -> dict:
    """Extract JSON-LD structured metadata."""
    import json as _json
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(script.string or "")
            if isinstance(data, dict) and "@graph" in data:
                for item in data["@graph"]:
                    if isinstance(item, dict) and (
                        "headline" in item or "author" in item
                    ):
                        return item
            if isinstance(data, list):
                data = data[0]
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _extract_title(soup: BeautifulSoup, json_ld: dict) -> str:
    """Extract title — JSON-LD → og:title → h1 → <title>"""
    t = json_ld.get("headline", "").strip()
    if t:
        return t
    og = soup.find("meta", property="og:title")
    if og and og.get("content", "").strip():
        return og["content"].strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)
        for sep in [" | ", " – ", " - ", " · "]:
            if sep in title:
                return title.split(sep)[0].strip()
        return title
    return "Unknown Title"


def _extract_author(soup: BeautifulSoup, json_ld: dict) -> str:
    """
    Extract author name.
    Priority: JSON-LD → meta → rel=author → class selectors
    """
    af = json_ld.get("author")
    if af:
        if isinstance(af, str) and af.strip():
            return af.strip()
        if isinstance(af, dict):
            n = af.get("name", "").strip()
            if n:
                return n
        if isinstance(af, list):
            names = []
            for a in af:
                if isinstance(a, dict):
                    n = a.get("name", "").strip()
                    if n:
                        names.append(n)
                elif isinstance(a, str) and a.strip():
                    names.append(a.strip())
            if names:
                return ", ".join(names)

    meta = soup.find("meta", attrs={"name": "author"})
    if meta and meta.get("content", "").strip():
        return meta["content"].strip()

    rel = soup.find(attrs={"rel": "author"})
    if rel:
        text = rel.get_text(strip=True)
        if text and len(text) < 80:
            return text

    for tag_name in ["a", "span", "div", "p"]:
        for hint in ["author", "byline", "writer", "creator"]:
            el = soup.find(
                tag_name,
                class_=lambda c: c and hint in " ".join(c).lower()
            )
            if el:
                text = el.get_text(strip=True)
                text = re.sub(r'^[Bb]y\s+', '', text).strip()
                if text and len(text) < 80:
                    return text

    return "Unknown Author"


def _extract_date(soup: BeautifulSoup, json_ld: dict) -> str:
    """Extract publication date → YYYY-MM-DD or 'Unknown'."""
    def try_parse(s) -> Optional[str]:
        if not s:
            return None
        try:
            return date_parser.parse(
                str(s), fuzzy=True
            ).strftime("%Y-%m-%d")
        except Exception:
            return None

    for field in ("datePublished", "dateCreated", "dateModified"):
        r = try_parse(json_ld.get(field, ""))
        if r:
            return r

    for attr, val in [
        ("property", "article:published_time"),
        ("property", "og:article:published_time"),
        ("name",     "date"),
        ("name",     "publish-date"),
        ("itemprop", "datePublished"),
    ]:
        tag = soup.find("meta", attrs={attr: val})
        if tag and tag.get("content"):
            r = try_parse(tag["content"])
            if r:
                return r

    for time_tag in soup.find_all("time"):
        r = try_parse(
            time_tag.get("datetime") or time_tag.get_text(strip=True)
        )
        if r:
            return r

    return "Unknown"


# ============================================================
# Medium — Clap Count
# ============================================================

def _scrape_clap_count(page: Page) -> int:
    """Scrape clap count from Medium article page."""
    print(f"[CLAPS] Scraping clap count...")

    clap_selectors = [
        "div.pw-multi-vote-count",
        "button[data-testid='meterButton'] span",
        "span.pw-upvote-count",
        "[data-testid='clapButton'] span",
        "div[class*='clap'] span",
        "button[aria-label*='clap'] span",
    ]

    for selector in clap_selectors:
        try:
            el = page.query_selector(selector)
            if el:
                text = el.inner_text().strip()
                count = _parse_count_text(text)
                if count > 0:
                    print(f"[CLAPS] Found via '{selector}': {count:,}")
                    return count
        except Exception:
            continue

    try:
        body_text = page.inner_text("body")
        patterns = [
            r'([\d,]+\.?\d*[KkMm]?)\s*[Cc]lap',
            r'[Cc]lap[s]?\s*\n?\s*([\d,]+\.?\d*[KkMm]?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, body_text)
            if match:
                count = _parse_count_text(match.group(1))
                if count > 0:
                    print(f"[CLAPS] Found via scan: {count:,}")
                    return count
    except Exception:
        pass

    print(f"[CLAPS] Not found → 0")
    return 0


# ============================================================
# Medium — Article Scraper (Playwright)
# ============================================================

def _scrape_medium_article(url: str) -> dict:
    """
    Scrape Medium article using Playwright.
    Extracts: content, claps, metadata.
    Author profile scraped separately in Phase 2.
    """
    print(f"\n{'='*60}")
    print(f"[ARTICLE] {url}")
    print(f"{'='*60}")

    domain = _get_domain(url)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ]
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        page = context.new_page()

        # Block images/media/fonts for speed
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ["image", "media", "font"]
            else route.continue_()
        )

        try:
            print(f"[PLAYWRIGHT] Loading article...")
            _navigate(page, url)
            _scroll_page(page)
            print(f"[PLAYWRIGHT] Page: {len(page.content()):,} bytes")

            # Claps — must scrape BEFORE navigating away
            clap_count = _scrape_clap_count(page)

            # Metadata from rendered HTML
            html     = page.content()
            soup     = BeautifulSoup(html, "lxml")
            json_ld  = _extract_json_ld(soup)
            title    = _extract_title(soup, json_ld)
            author   = _extract_author(soup, json_ld)
            pub_date = _extract_date(soup, json_ld)

            print(f"\n[META] Title  : {title[:70]}")
            print(f"[META] Author : {author}")
            print(f"[META] Date   : {pub_date}")
            print(f"[META] Claps  : {clap_count:,}")

            # Full page text
            print(f"\n[CONTENT] Capturing text...")
            try:
                raw_text = page.inner_text("body")
            except Exception:
                raw_text = page.locator("body").inner_text()

            print(f"[CONTENT] Raw: {len(raw_text):,} chars")

        except Exception as e:
            print(f"[ERROR] Scrape failed: {e}")
            browser.close()
            return _error_result(url, str(e))

        browser.close()

    # Clean + NLP
    print(f"\n[CONTENT] Cleaning...")
    content    = _clean_raw_text(raw_text)
    word_count = len(content.split())
    print(f"[CONTENT] Words: {word_count:,}")

    language       = detect_language(content)
    has_disclaimer = _detect_medical_disclaimer(content)

    print(f"[NLP] Language   : {language}")
    print(f"[NLP] Disclaimer : {has_disclaimer}")

    print(f"[NLP] Extracting tags...")
    tags = extract_tags(content)
    print(f"[NLP] Tags       : {tags}")

    print(f"[NLP] Chunking...")
    chunks = chunk_text(content)
    print(f"[NLP] Chunks     : {len(chunks)}")

    time.sleep(settings.request_delay)

    return {
        "source_url":     url,
        "source_type":    "blog",
        "title":          title,
        "author":         author,
        "published_date": pub_date,
        "language":       language,
        "region":         "Unknown",
        "topic_tags":     tags,
        "trust_score":    None,
        "content_chunks": chunks,
        "metadata": {
            "domain":                 domain,
            "has_medical_disclaimer": has_disclaimer,
            "word_count":             word_count,
            "chunk_count":            len(chunks),
            "clap_count":             clap_count,
            "author_profile": {
                "followers":     0,
                "bio":           "",
                "article_count": 0,
                "profile_url":   "",
            },
            "scraped_at": _now_iso(),
        },
    }


# ============================================================
# Nature.com — Springer API + requests
# ============================================================

def _extract_nature_doi(article_url: str) -> str:
    """
    Extract DOI from Nature.com URL.
    "https://www.nature.com/articles/d41586-026-01149-9"
      → "10.1038/d41586-026-01149-9"
    """
    match = re.search(
        r'nature\.com/articles/([a-z0-9-]+)',
        article_url.rstrip("/")
    )
    if match:
        return f"10.1038/{match.group(1)}"
    return ""


def _fetch_springer_metadata(doi: str) -> dict:
    """
    Fetch article metadata from Springer Nature Meta API.

    Returns: title, authors, affiliations, journal,
             date, abstract, subjects, DOI, OA flag.
    """
    api_key = settings.springer_meta_api_key
    if not api_key:
        print(f"[META API] ⚠ SPRINGER_META_API_KEY not set in .env")
        return {}

    print(f"[META API] DOI: {doi}")

    try:
        response = requests.get(
            SPRINGER_META_URL,
            params={"doi": doi, "api_key": api_key},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        if not records:
            print(f"[META API] ⚠ No records for DOI: {doi}")
            return {}

        record = records[0]

        # Authors + affiliations
        creators     = record.get("creators", [])
        authors_list = []
        affiliations = []

        for creator in creators:
            name = creator.get("creator", "").strip()
            if name:
                authors_list.append(name)
            affil = creator.get("affiliation", "")
            if affil and affil not in affiliations:
                affiliations.append(affil)

        # Subjects as topic tags
        subjects = [
            s.get("term", "").strip()
            for s in record.get("subjects", [])
            if s.get("term", "").strip()
        ]

        # Publication date
        pub_date = _parse_springer_date(record)

        # Open Access flag
        is_oa = str(record.get("openaccess", "false")).lower() == "true"

        result = {
            "title":          record.get("title", "Unknown Title"),
            "author":         ", ".join(authors_list) or "Unknown Author",
            "authors_list":   authors_list,
            "affiliations":   affiliations,
            "journal":        record.get("publicationName", "Nature"),
            "pub_date":       pub_date,
            "abstract":       record.get("abstract", ""),
            "subjects":       subjects,
            "doi":            doi,
            "is_open_access": is_oa,
        }

        print(f"[META API] ✅ Title      : {result['title'][:60]}")
        print(f"[META API] ✅ Authors    : {result['author'][:60]}")
        print(f"[META API] ✅ Journal    : {result['journal']}")
        print(f"[META API] ✅ Date       : {result['pub_date']}")
        print(f"[META API] ✅ OA         : {is_oa}")
        print(f"[META API] ✅ Subjects   : {subjects[:3]}")
        print(f"[META API] ✅ Affiliations: {affiliations[:2]}")

        return result

    except Exception as e:
        print(f"[META API] ⚠ Failed: {e}")
        return {}


def _fetch_springer_fulltext(doi: str) -> str:
    """
    Fetch full text from Springer Open Access API.
    Only works for OA articles.
    """
    api_key = settings.springer_oa_api_key
    if not api_key:
        print(f"[OA API] ⚠ SPRINGER_OA_API_KEY not set in .env")
        return ""

    print(f"[OA API] Fetching full text: {doi}")

    try:
        response = requests.get(
            SPRINGER_OA_URL,
            params={"doi": doi, "api_key": api_key},
            headers={"Accept": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        records = data.get("records", [])
        if not records:
            print(f"[OA API] ⚠ No OA records — not Open Access")
            return ""

        record   = records[0]
        fulltext = (
            record.get("bodyText", "")
            or record.get("body", "")
            or record.get("fullText", "")
        )

        if fulltext:
            # Strip HTML if present
            if "<" in fulltext:
                soup = BeautifulSoup(fulltext, "lxml")
                fulltext = soup.get_text(separator="\n", strip=True)
            print(f"[OA API] ✅ Full text: {len(fulltext.split()):,} words")
            return fulltext

        print(f"[OA API] ⚠ Empty body in OA response")
        return ""

    except Exception as e:
        print(f"[OA API] ⚠ Failed: {e}")
        return ""


def _fetch_nature_text_requests(article_url: str) -> str:
    """
    Fetch Nature article text via requests + BS4.

    Why this works when Playwright doesn't:
    → Nature uses Akamai bot detection on browser automation
    → requests has a normal TLS fingerprint
    → Article body is server-side rendered (no JS needed)
    """
    print(f"[BS4] Fetching via requests...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language":           "en-US,en;q=0.9",
        "Accept-Encoding":           "gzip, deflate, br",
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":            "document",
        "Sec-Fetch-Mode":            "navigate",
        "Sec-Fetch-Site":            "none",
        "Cache-Control":             "max-age=0",
    }

    try:
        response = requests.get(
            article_url,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()

        size = len(response.text)
        print(f"[BS4] Status: {response.status_code} | Size: {size:,} chars")

        # CAPTCHA check
        if size < 5000:
            print(f"[BS4] ⚠ Response too small — likely blocked")
            return ""

        soup = BeautifulSoup(response.text, "lxml")

        # Remove noise
        for tag in soup.find_all([
            "script", "style", "nav", "footer",
            "header", "aside", "noscript", "figure",
        ]):
            tag.decompose()

        # Nature content selectors (priority order)
        content_selectors = [
            "div.article__body",
            "div[data-article-body='true']",
            "div.c-article-body",
            "section.article__body",
            "article",
            "main",
        ]

        for selector in content_selectors:
            tag = soup.select_one(selector)
            if tag:
                paragraphs = tag.find_all("p")
                text = "\n".join(
                    p.get_text(separator=" ", strip=True)
                    for p in paragraphs
                    if len(p.get_text(strip=True)) > 30
                )
                if len(text.split()) > 100:
                    print(f"[BS4] ✅ '{selector}': {len(text.split()):,} words")
                    return text

        # Fallback: all paragraphs
        paragraphs = soup.find_all("p")
        text = "\n".join(
            p.get_text(separator=" ", strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 30
        )
        if text:
            print(f"[BS4] ✅ Fallback: {len(text.split()):,} words")
        else:
            print(f"[BS4] ⚠ No text extracted")

        return text

    except Exception as e:
        print(f"[BS4] ⚠ Failed: {e}")
        return ""


def _parse_springer_date(record: dict) -> str:
    """Parse date from Springer API record."""
    for field in ["publicationDate", "onlineDate", "printDate", "coverDate"]:
        date_str = record.get(field, "")
        if date_str:
            try:
                if len(date_str) == 7:
                    date_str += "-01"
                return date_parser.parse(date_str, fuzzy=True).strftime("%Y-%m-%d")
            except Exception:
                continue
    return "Unknown"


def _parse_affiliation_text(affiliation: str) -> dict:
    """
    Parse affiliation string into structured fields.

    Input:
      "Department of Chemical Engineering, University of Cambridge, UK"
    Output:
      {department, institution, region}
    """
    result = {"department": "", "institution": "", "region": ""}
    parts  = [p.strip() for p in affiliation.split(",") if p.strip()]

    dept_kw = [
        "department", "faculty", "school of", "division",
        "centre for", "center for", "institute of",
        "laboratory", "programme", "program",
    ]
    inst_kw = [
        "university", "institute", "college", "hospital",
        "foundation", "academy", "research center",
        "technische", "école",
    ]

    for part in parts:
        part_lower = part.lower()
        if not result["department"] and any(k in part_lower for k in dept_kw):
            result["department"] = part
        if not result["institution"] and any(k in part_lower for k in inst_kw):
            result["institution"] = part

    if parts:
        last    = parts[-1]
        cleaned = re.sub(r'\b[A-Z]{1,2}\d+\s*[A-Z]*\d*\b', '', last)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().strip(",")
        if cleaned and len(cleaned) <= 30:
            result["region"] = cleaned

    return result


def _build_nature_author_profile(metadata: dict) -> dict:
    """
    Build author profile from Springer API metadata.

    Nature authors have verified institutional affiliations
    submitted during peer review — high credibility signal.
    """
    affiliations = metadata.get("affiliations", [])
    authors_list = metadata.get("authors_list", [])

    institution = ""
    department  = ""
    region      = ""

    if affiliations:
        parsed      = _parse_affiliation_text(affiliations[0])
        institution = parsed.get("institution", "")
        department  = parsed.get("department", "")
        region      = parsed.get("region", "")

    bio = " | ".join(affiliations) if affiliations else ""

    return {
        "followers":    0,
        "bio":          bio,
        "article_count": 0,
        "profile_url":  "",
        "institution":  institution,
        "department":   department,
        "region":       region,
        "orcid":        "",
        "all_authors":  authors_list,
        "affiliations": affiliations,
    }


def _scrape_nature_article_full(url: str) -> dict:
    """
    Complete Nature.com article scraping.

    Pipeline:
      1. Extract DOI from URL
      2. Springer Meta API → metadata + author affiliations
      3. Springer OA API → full text (if Open Access)
      4. requests + BS4 → full text fallback
      5. NLP pipeline → tags (from API subjects + KeyBERT)
      6. Return structured dict

    No Playwright used — avoids CAPTCHA entirely.
    """
    print(f"\n{'='*60}")
    print(f"[NATURE] {url}")
    print(f"{'='*60}")

    domain = _get_domain(url)

    # ── Step 1: DOI ────────────────────────────────────────
    doi = _extract_nature_doi(url)
    if not doi:
        print(f"[NATURE] ⚠ Cannot extract DOI")
        return _error_result(url, "Cannot extract DOI from URL")

    print(f"[NATURE] DOI: {doi}")

    # ── Step 2: Metadata via Meta API ─────────────────────
    print(f"\n[STEP 1] Springer Meta API...")
    metadata = _fetch_springer_metadata(doi)

    if not metadata:
        # Fallback metadata
        print(f"[NATURE] ⚠ Meta API failed — using minimal metadata")
        metadata = {
            "title":          "Unknown Title",
            "author":         "Unknown Author",
            "authors_list":   [],
            "affiliations":   [],
            "journal":        "Nature",
            "pub_date":       "Unknown",
            "abstract":       "",
            "subjects":       [],
            "doi":            doi,
            "is_open_access": False,
        }

    # ── Step 3: Full text ──────────────────────────────────
    print(f"\n[STEP 2] Fetching full text...")
    full_text = ""

    # OA API first (if article is open access)
    if metadata.get("is_open_access"):
        print(f"[NATURE] Article is Open Access → trying OA API")
        full_text = _fetch_springer_fulltext(doi)

    # Fallback: requests + BS4
    if not full_text or len(full_text.split()) < 100:
        print(f"[NATURE] OA API insufficient → requests + BS4")
        full_text = _fetch_nature_text_requests(url)

    # Last resort: abstract only
    if not full_text or len(full_text.split()) < 50:
        print(f"[NATURE] ⚠ Using abstract only")
        full_text = metadata.get("abstract", "")

    word_count = len(full_text.split())
    print(f"\n[NATURE] Content: {word_count:,} words")
    print(f"[NATURE] Preview: {full_text[:200]}...")

    # ── Step 4: NLP Pipeline ──────────────────────────────
    print(f"\n[NLP] Processing...")

    language = detect_language(full_text)
    print(f"[NLP] Language   : {language}")

    has_disclaimer = _detect_medical_disclaimer(full_text)
    print(f"[NLP] Disclaimer : {has_disclaimer}")

    # Topic tags: Springer subjects + KeyBERT supplement
    springer_tags = metadata.get("subjects", [])
    if len(springer_tags) >= 4:
        tags = springer_tags[:6]
        print(f"[NLP] Tags (API)     : {tags}")
    else:
        print(f"[NLP] Supplementing with KeyBERT...")
        keybert_tags = extract_tags(full_text)
        tags = list(dict.fromkeys(springer_tags + keybert_tags))[:6]
        print(f"[NLP] Tags (combined): {tags}")

    print(f"[NLP] Chunking...")
    chunks = chunk_text(full_text)
    print(f"[NLP] Chunks     : {len(chunks)}")

    # ── Author profile from API ────────────────────────────
    author_profile = _build_nature_author_profile(metadata)

    time.sleep(settings.request_delay)

    return {
        "source_url":     url,
        "source_type":    "blog",
        "title":          metadata["title"],
        "author":         metadata["author"],
        "published_date": metadata["pub_date"],
        "language":       language,
        "region":         author_profile.get("region", "Unknown"),
        "topic_tags":     tags,
        "trust_score":    None,
        "content_chunks": chunks,
        "metadata": {
            "domain":                 domain,
            "has_medical_disclaimer": has_disclaimer,
            "word_count":             word_count,
            "chunk_count":            len(chunks),
            "clap_count":             0,
            "journal":                metadata.get("journal", ""),
            "doi":                    doi,
            "is_open_access":         metadata.get("is_open_access", False),
            "abstract":               metadata.get("abstract", "")[:500],
            "author_profile":         author_profile,
            "scraped_at":             _now_iso(),
        },
    }


# ============================================================
# Phase 1 — Route by Domain
# ============================================================

# ============================================================
# HARVARD HEALTH BLOG — requests + BS4
# No Playwright needed — server-side rendered
# ============================================================

HARVARD_HEALTH_DOMAIN = "health.harvard.edu"

def _extract_harvard_content(soup: BeautifulSoup) -> str:
    """
    Extract article body text from Harvard Health page.
    Removes nav, ads, subscription prompts.
    """
    # Remove noise
    for tag in soup.find_all([
        "script", "style", "nav", "footer",
        "header", "aside", "noscript",
    ]):
        tag.decompose()

    for cls in [
        "paywall", "subscribe", "subscription",
        "newsletter", "related", "sidebar",
        "advertisement", "promo", "cta",
    ]:
        for el in soup.find_all(
            class_=re.compile(cls, re.IGNORECASE)
        ):
            el.decompose()

    # Content selectors in priority order
    content_selectors = [
        "div.article-content",
        "div.entry-content",
        "div.post-content",
        "div.blog-content",
        "div[class*='article-body']",
        "div[class*='content-body']",
        "article",
        "main",
    ]

    for selector in content_selectors:
        tag = soup.select_one(selector)
        if tag:
            paragraphs = tag.find_all("p")
            text = "\n".join(
                p.get_text(separator=" ", strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 30
            )
            if len(text.split()) > 100:
                print(f"[HARVARD] Content via '{selector}'")
                return text

    # Fallback: all paragraphs
    paragraphs = soup.find_all("p")
    return "\n".join(
        p.get_text(separator=" ", strip=True)
        for p in paragraphs
        if len(p.get_text(strip=True)) > 30
    )

def _fetch_harvard_article(url: str) -> dict:
    """
    Scrape Harvard Health Blog article using requests + BS4.

    Why requests works here:
    → Harvard Health renders content server-side
    → No JavaScript needed for article body
    → requests has cleaner TLS fingerprint than Playwright
    → Playwright triggers reCAPTCHA, requests does not

    Returns full article dict matching schema.
    """
    print(f"\n{'='*60}")
    print(f"[HARVARD] {url}")
    print(f"{'='*60}")

    domain = _get_domain(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language":           "en-US,en;q=0.9",
        "Accept-Encoding":           "gzip, deflate, br",
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":            "document",
        "Sec-Fetch-Mode":            "navigate",
        "Sec-Fetch-Site":            "none",
        "Cache-Control":             "max-age=0",
    }

    # ── Step 1: Fetch article ──────────────────────────────
    print(f"[HARVARD] Fetching article...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        print(f"[HARVARD] Status: {response.status_code} | Size: {len(response.text):,}")
    except Exception as e:
        print(f"[HARVARD] ⚠ Fetch failed: {e}")
        return _error_result(url, str(e))

    soup = BeautifulSoup(response.text, "lxml")

    # ── Step 2: Metadata ───────────────────────────────────
    print(f"[HARVARD] Extracting metadata...")
    title, authors_list, author_urls, pub_date, section = (
        _extract_harvard_metadata(soup)
    )

    print(f"[HARVARD] ✅ Title   : {title[:60]}")
    print(f"[HARVARD] ✅ Authors : {authors_list}")
    print(f"[HARVARD] ✅ Date    : {pub_date}")
    print(f"[HARVARD] ✅ Section : {section}")
    print(f"[HARVARD] ✅ Links   : {author_urls}")

    # ── Step 3: Article content ────────────────────────────
    print(f"[HARVARD] Extracting content...")
    content = _extract_harvard_content(soup)
    word_count = len(content.split())
    print(f"[HARVARD] ✅ Words   : {word_count:,}")
    print(f"[HARVARD] Preview   : {content[:200]}...")

    # ── Step 4: Author profiles ────────────────────────────
    # Fetch each author page via requests
    print(f"\n[HARVARD] Fetching author profiles...")
    author_profile = _fetch_harvard_author_profiles(
        author_urls, headers
    )

    # ── Step 5: NLP ────────────────────────────────────────
    print(f"\n[NLP] Processing...")
    language       = detect_language(content)
    has_disclaimer = _detect_medical_disclaimer(content)

    print(f"[NLP] Language   : {language}")
    print(f"[NLP] Disclaimer : {has_disclaimer}")

    print(f"[NLP] Extracting tags...")
    tags = extract_tags(content)
    print(f"[NLP] Tags       : {tags}")

    print(f"[NLP] Chunking...")
    chunks = chunk_text(content)
    print(f"[NLP] Chunks     : {len(chunks)}")

    time.sleep(settings.request_delay)

    return {
        "source_url":     url,
        "source_type":    "blog",
        "title":          title,
        "author":         ", ".join(authors_list) if authors_list else "Unknown Author",
        "published_date": pub_date,
        "language":       language,
        "region":         "US",  # Harvard Health is US-based
        "topic_tags":     tags,
        "trust_score":    None,
        "content_chunks": chunks,
        "metadata": {
            "domain":                 domain,
            "has_medical_disclaimer": has_disclaimer,
            "word_count":             word_count,
            "chunk_count":            len(chunks),
            "clap_count":             0,
            "section":                section,
            "author_profile":         author_profile,
            "scraped_at":             _now_iso(),
        },
    }


def _extract_harvard_metadata(soup: BeautifulSoup) -> tuple:
    """
    Extract metadata from Harvard Health article page.

    Uses meta tags (most reliable for Harvard Health):
      article:author        → author name
      article:published_time → publish date
      article:section       → topic/section

    Also extracts author links for profile fetching.

    Returns:
        (title, authors_list, author_urls, pub_date, section)
    """
    # ── Title ──────────────────────────────────────────────
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
        # Remove " - Harvard Health" suffix
        for suffix in [" - Harvard Health", " | Harvard Health"]:
            if title.endswith(suffix):
                title = title[: -len(suffix)].strip()

    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else "Unknown Title"

    # ── Authors from meta ──────────────────────────────────
    # Harvard puts authors in article:author meta tag
    authors_list = []
    for meta in soup.find_all("meta", property="article:author"):
        name = meta.get("content", "").strip()
        if name and name not in authors_list:
            authors_list.append(name)

    # ── Author links from HTML ─────────────────────────────
    # These lead to the author profile pages
    author_urls = []
    for link in soup.find_all("a", href=re.compile(r"/authors/")):
        href = link.get("href", "")
        if href and href not in author_urls:
            # Ensure full URL
            if href.startswith("/"):
                href = f"https://www.health.harvard.edu{href}"
            author_urls.append(href)

        # Also capture names from links if not in meta
        name = link.get_text(strip=True)
        if name and name not in authors_list and len(name) > 2:
            authors_list.append(name)

    # Deduplicate authors
    authors_list = list(dict.fromkeys(authors_list))

    # ── Publication date ───────────────────────────────────
    pub_date = ""
    date_meta = soup.find(
        "meta", property="article:published_time"
    )
    if date_meta and date_meta.get("content"):
        try:
            pub_date = date_parser.parse(
                date_meta["content"], fuzzy=True
            ).strftime("%Y-%m-%d")
        except Exception:
            pass

    if not pub_date:
        # Try time tag
        time_tag = soup.find("time")
        if time_tag:
            try:
                pub_date = date_parser.parse(
                    time_tag.get("datetime", "") or time_tag.get_text(),
                    fuzzy=True,
                ).strftime("%Y-%m-%d")
            except Exception:
                pass

    if not pub_date:
        pub_date = "Unknown"

    # ── Section / Topic ────────────────────────────────────
    section = ""
    section_meta = soup.find("meta", property="article:section")
    if section_meta:
        section = section_meta.get("content", "").strip()

    return title, authors_list, author_urls, pub_date, section


def _extract_harvard_metadata(soup: BeautifulSoup) -> tuple:
    """
    Extract metadata from Harvard Health article page.
    Fixed: article:author meta property checked first.
    """
    # ── Title ──────────────────────────────────────────────
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
        for suffix in [" - Harvard Health", " | Harvard Health"]:
            if title.endswith(suffix):
                title = title[: -len(suffix)].strip()

    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else "Unknown Title"

    # ── Authors ────────────────────────────────────────────
    # Priority 1: article:author meta property (most reliable)
    authors_list = []
    for meta in soup.find_all("meta", property="article:author"):
        name = (meta.get("content") or "").strip()
        if name and name not in authors_list:
            authors_list.append(name)

    # Priority 2: rel=author links
    if not authors_list:
        for link in soup.find_all("a", rel="author"):
            name = link.get_text(strip=True)
            if name and name not in authors_list and len(name) > 2:
                authors_list.append(name)

    # Priority 3: /authors/ links
    # Filter carefully — skip "See Full Bio", "View all posts by..."
    if not authors_list:
        SKIP_TEXTS = {
            "see full bio", "view all posts",
            "view all", "next", "previous",
            "back", "more", "read more",
        }
        seen_hrefs = set()
        for link in soup.find_all("a", href=re.compile(r"/authors/")):
            href = link.get("href", "")
            name = link.get_text(strip=True)
            name_lower = name.lower()

            # Skip navigation/utility links
            if any(skip in name_lower for skip in SKIP_TEXTS):
                continue

            # Skip duplicates by href
            if href in seen_hrefs:
                continue

            # Skip very short or very long names
            if len(name) < 3 or len(name) > 80:
                continue

            seen_hrefs.add(href)
            if name not in authors_list:
                authors_list.append(name)

    # Deduplicate preserving order
    authors_list = list(dict.fromkeys(authors_list))

    # ── Author URLs (for profile fetching) ─────────────────
    # Only unique profile URLs — skip "See Full Bio" duplicates
    seen_hrefs  = set()
    author_urls = []
    SKIP_TEXTS  = {
        "see full bio", "view all posts",
        "view all", "next", "previous",
    }

    for link in soup.find_all("a", href=re.compile(r"/authors/")):
        href = link.get("href", "")
        name = link.get_text(strip=True).lower()

        if any(skip in name for skip in SKIP_TEXTS):
            continue
        if href in seen_hrefs:
            continue
        if not href:
            continue

        if href.startswith("/"):
            href = f"https://www.health.harvard.edu{href}"

        seen_hrefs.add(href)
        author_urls.append(href)

    # ── Publication date ───────────────────────────────────
    pub_date = ""
    date_meta = soup.find("meta", property="article:published_time")
    if date_meta and date_meta.get("content"):
        try:
            pub_date = date_parser.parse(
                date_meta["content"], fuzzy=True
            ).strftime("%Y-%m-%d")
        except Exception:
            pass

    if not pub_date:
        time_tag = soup.find("time")
        if time_tag:
            try:
                pub_date = date_parser.parse(
                    time_tag.get("datetime", "")
                    or time_tag.get_text(),
                    fuzzy=True,
                ).strftime("%Y-%m-%d")
            except Exception:
                pass

    pub_date = pub_date or "Unknown"

    # ── Section ────────────────────────────────────────────
    section = ""
    section_meta = soup.find("meta", property="article:section")
    if section_meta:
        section = section_meta.get("content", "").strip()

    print(f"[HARVARD META] Authors   : {authors_list}")
    print(f"[HARVARD META] Author URLs: {author_urls}")
    print(f"[HARVARD META] Date      : {pub_date}")
    print(f"[HARVARD META] Section   : {section}")

    return title, authors_list, author_urls, pub_date, section

def _fetch_harvard_author_profiles(
    author_urls: list,
    headers: dict,
) -> dict:
    """
    Fetch author profile pages from Harvard Health.

    Each author page contains:
    - Full name
    - Title/credentials (e.g., "MD", "PhD")
    - Affiliation (e.g., "Harvard Medical School")
    - Bio paragraph
    - Number of articles

    Returns combined author profile dict for trust scoring.
    """
    if not author_urls:
        print(f"[HARVARD AUTH] No author URLs to fetch")
        return {
            "followers":     0,
            "bio":           "",
            "article_count": 0,
            "profile_url":   "",
            "institution":   "Harvard Health Publishing",
            "department":    "",
            "region":        "US",
            "credentials":   [],
            "all_authors":   [],
        }

    all_bios        = []
    all_credentials = []
    all_names       = []
    institution     = "Harvard Health Publishing"
    article_count   = 0

    for author_url in author_urls[:3]:  # Cap at 3 authors
        print(f"[HARVARD AUTH] Fetching: {author_url}")

        # Polite delay between author page requests
        time.sleep(1.5)

        try:
            response = requests.get(
                author_url,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()

            author_soup = BeautifulSoup(response.text, "lxml")
            author_data = _parse_harvard_author_page(author_soup, author_url)

            if author_data.get("name"):
                all_names.append(author_data["name"])
            if author_data.get("bio"):
                all_bios.append(author_data["bio"])
            if author_data.get("credentials"):
                all_credentials.extend(author_data["credentials"])
            if author_data.get("article_count", 0) > article_count:
                article_count = author_data["article_count"]

            print(f"[HARVARD AUTH] ✅ Name        : {author_data.get('name', 'N/A')}")
            print(f"[HARVARD AUTH] ✅ Credentials : {author_data.get('credentials', [])}")
            print(f"[HARVARD AUTH] ✅ Bio preview  : {author_data.get('bio', '')[:80]}")
            print(f"[HARVARD AUTH] ✅ Articles     : {author_data.get('article_count', 0)}")

        except Exception as e:
            print(f"[HARVARD AUTH] ⚠ Failed for {author_url}: {e}")
            continue

    # Build combined profile
    combined_bio = " | ".join(all_bios) if all_bios else ""
    # Remove duplicate credentials
    unique_creds = list(dict.fromkeys(all_credentials))

    return {
        "followers":     0,       # Not applicable
        "bio":           combined_bio,
        "article_count": article_count,
        "profile_url":   author_urls[0] if author_urls else "",
        "institution":   institution,
        "department":    "Harvard Medical School",
        "region":        "US",
        "credentials":   unique_creds,
        "all_authors":   all_names,
    }


def _parse_harvard_author_page(
    soup: BeautifulSoup,
    author_url: str,
) -> dict:
    """
    Parse a single Harvard Health author profile page.

    Author page structure:
      - Author name in <h1>
      - Title/credentials below name
      - Bio paragraph
      - List of published articles (for article count)
    """
    result = {
        "name":          "",
        "bio":           "",
        "credentials":   [],
        "article_count": 0,
    }

    # ── Name ───────────────────────────────────────────────
    h1 = soup.find("h1")
    if h1:
        result["name"] = h1.get_text(strip=True)

    # Remove from og:title as fallback
    if not result["name"]:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            name = og["content"].replace("- Harvard Health", "").strip()
            result["name"] = name

    # ── Credentials ────────────────────────────────────────
    # Harvard author pages list credentials near the name
    CREDENTIAL_PATTERNS = [
        r'\bMD\b', r'\bPhD\b', r'\bPh\.D\b',
        r'\bDO\b', r'\bMPH\b', r'\bRN\b',
        r'\bNP\b', r'\bPA\b', r'\bDDS\b',
        r'\bPsyD\b', r'\bLCSW\b', r'\bMSW\b',
        r'\bFACC\b', r'\bFACS\b',  # Fellowship credentials
    ]

    # Check author name + page text for credentials
    page_text = soup.get_text()
    name_text = result["name"]
    search_text = f"{name_text} {page_text[:2000]}"

    for pattern in CREDENTIAL_PATTERNS:
        if re.search(pattern, search_text):
            cred = pattern.replace(r'\b', '').replace('\\', '')
            if cred not in result["credentials"]:
                result["credentials"].append(cred)

    # ── Bio ────────────────────────────────────────────────
    bio_selectors = [
        "div.author-bio",
        "div.author-description",
        "div[class*='author-about']",
        "div[class*='bio']",
        "section.about",
        "p.author-description",
    ]

    for selector in bio_selectors:
        el = soup.select_one(selector)
        if el:
            bio_text = el.get_text(separator=" ", strip=True)
            if len(bio_text) > 30:
                result["bio"] = bio_text
                break

    # Fallback: find substantial paragraphs after h1
    if not result["bio"]:
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            text = p.get_text(strip=True)
            # Bio paragraphs are usually > 50 chars
            # and don't look like navigation
            if (
                len(text) > 50
                and "harvard" in text.lower()
                or "editor" in text.lower()
                or "writer" in text.lower()
                or "research" in text.lower()
                or "journalist" in text.lower()
            ):
                result["bio"] = text
                break

    # ── Article count ──────────────────────────────────────
    # Count article links on author page
    article_links = soup.find_all(
        "a", href=re.compile(r"/blog/|/staying-healthy/|/diseases-and-conditions/")
    )
    result["article_count"] = len(article_links)

    return result

def _scrape_single_article(url: str) -> dict:
    """
    Phase 1: Route to correct scraper based on domain.

    medium.com       → Playwright
    nature.com       → Springer API + requests
    health.harvard.edu → requests + BS4
    """
    domain = _get_domain(url)

    if "nature.com" in domain:
        return _scrape_nature_article_full(url)

    if "health.harvard.edu" in domain:
        return _fetch_harvard_article(url)

    if "medium.com" in domain:
        return _scrape_medium_article(url)

    # Unknown → attempt Playwright
    print(f"[WARN] Unknown domain: {domain} → attempting Playwright")
    return _scrape_medium_article(url)

# ============================================================
# Phase 2 — Medium Author Profile
# ============================================================

def _find_followers_in_body(body_text: str) -> int:
    """
    Find follower count from raw body text.
    Handles "2.1K followers", "499 Followers", split lines.
    """
    if not body_text:
        return 0

    lines = [l.strip() for l in body_text.split("\n") if l.strip()]

    for i, line in enumerate(lines):
        line_lower = line.lower()

        if "follower" in line_lower:
            match = re.search(
                r'([\d]+\.?[\d]*\s*[KkMm]?)\s*[Ff]ollower',
                line
            )
            if match:
                count = _parse_count_text(match.group(1))
                if count > 0:
                    print(f"[FOLLOWERS] Matched: '{line.strip()}' → {count:,}")
                    return count

            # Number on previous line
            if i > 0:
                prev = lines[i - 1].strip()
                if re.match(r'^[\d,.]+\s*[KkMm]?$', prev, re.IGNORECASE):
                    count = _parse_count_text(prev)
                    if count > 0:
                        print(f"[FOLLOWERS] Prev-line: '{prev}' → {count:,}")
                        return count

            # Number on next line
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if re.match(r'^[\d,.]+\s*[KkMm]?$', nxt, re.IGNORECASE):
                    count = _parse_count_text(nxt)
                    if count > 0:
                        print(f"[FOLLOWERS] Next-line: '{nxt}' → {count:,}")
                        return count

        if line_lower in ("followers", "follower"):
            if i > 0:
                count = _parse_count_text(lines[i - 1].strip())
                if count > 0:
                    return count

    # Full text scan
    match = re.search(
        r'([\d]+\.?[\d]*\s*[KkMm]?)\s*[Ff]ollower',
        body_text
    )
    if match:
        count = _parse_count_text(match.group(1))
        if count > 0:
            print(f"[FOLLOWERS] Full-text: '{match.group(0)}' → {count:,}")
            return count

    return 0


def _find_bio_in_body(body_text: str, author_name: str) -> str:
    """
    Extract bio text from Medium About page.
    Looks for substantial text after the followers section.
    """
    lines = [l.strip() for l in body_text.split("\n") if l.strip()]

    skip = {
        "get app", "write", "sign up", "sign in",
        "home", "activity", "new", "about",
        "follow", "following", "·",
        "help", "status", "careers", "press", "blog",
        "privacy", "rules", "terms", "text to speech",
    }

    author_lower   = author_name.lower()
    bio_lines      = []
    past_followers = False

    for line in lines:
        line_lower = line.lower().strip()

        if "follower" in line_lower:
            past_followers = True
            continue

        if not past_followers:
            continue

        if line_lower in skip:
            continue
        if line_lower == author_lower:
            continue
        if line_lower.startswith("connect with"):
            continue
        if "medium member since" in line_lower:
            continue
        if re.match(
            r'^[\d,.]+\s*[KkMm]?\s*(followers?|following)$',
            line, re.IGNORECASE
        ):
            continue
        if re.match(r'^see all \(\d+\)$', line_lower):
            continue
        if len(line) < 15:
            continue
        if len(line.split()) <= 3 and len(line) < 40:
            continue

        bio_lines.append(line)
        if len(bio_lines) >= 3:
            break

    return " ".join(bio_lines)


def _find_article_count_in_body(body_text: str) -> int:
    """Extract story count from About page body text."""
    match = re.search(
        r'(\d+)\s*(stories|story|posts|articles)',
        body_text, re.IGNORECASE,
    )
    return int(match.group(1)) if match else 0


def _scrape_single_author(
    about_url: str,
    profile_url: str,
    author_name: str,
) -> dict:
    """
    Scrape a single Medium author About page.
    Opens FRESH browser per author — avoids Medium rate limiting.
    Retries once on failure or short response.
    """
    default = {
        "followers":     0,
        "bio":           "",
        "article_count": 0,
        "profile_url":   profile_url,
    }

    for attempt in range(2):
        if attempt > 0:
            print(f"[AUTHOR] Retry attempt {attempt + 1}...")
            time.sleep(3)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                    ]
                )
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 800},
                    locale="en-US",
                )
                page = context.new_page()

                try:
                    try:
                        page.goto(
                            about_url,
                            wait_until="networkidle",
                            timeout=20000,
                        )
                    except PWTimeout:
                        page.goto(
                            about_url,
                            wait_until="domcontentloaded",
                            timeout=15000,
                        )

                    page.wait_for_timeout(2000)

                    body_text = page.inner_text("body")
                    print(f"[AUTHOR] Page text: {len(body_text)} chars")

                    if len(body_text) < 300:
                        print(f"[AUTHOR] ⚠ Too short → likely blocked")
                        browser.close()
                        if attempt == 0:
                            continue
                        else:
                            return default

                    followers     = _find_followers_in_body(body_text)
                    bio           = _find_bio_in_body(body_text, author_name)
                    article_count = _find_article_count_in_body(body_text)

                    print(f"[AUTHOR] ✅ Followers : {followers:,}")
                    print(f"[AUTHOR] ✅ Bio       : {bio[:80]}..." if bio else "[AUTHOR] ⚠ Bio: (empty)")
                    print(f"[AUTHOR] ✅ Articles  : {article_count}")

                    browser.close()

                    return {
                        "followers":     followers,
                        "bio":           bio,
                        "article_count": article_count,
                        "profile_url":   profile_url,
                    }

                except Exception as e:
                    print(f"[AUTHOR] ⚠ Page error: {e}")
                    browser.close()
                    if attempt == 0:
                        continue
                    return default

        except Exception as e:
            print(f"[AUTHOR] ⚠ Browser error: {e}")
            if attempt == 0:
                continue
            return default

    return default


def _scrape_all_author_profiles(results: list) -> list:
    """
    Phase 2: Route author scraping by domain.

    medium.com         → fresh browser per author
    nature.com         → already done via API in Phase 1
    health.harvard.edu → already done via requests in Phase 1
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Author Profile Scraping")
    print(f"{'='*60}")

    total = len(results)

    for i, result in enumerate(results):
        domain      = result["metadata"].get("domain", "")
        article_url = result.get("source_url", "")
        author_name = result.get("author", "")

        print(f"\n[{i+1}/{total}] Domain: {domain} | {author_name[:40]}")

        if "nature.com" in domain:
            prof = result["metadata"].get("author_profile", {})
            print(f"[AUTHOR] Nature.com: API data ✅")
            print(f"[AUTHOR] Institution: {prof.get('institution', 'N/A')}")
            print(f"[AUTHOR] Region     : {prof.get('region', 'N/A')}")

        elif "health.harvard.edu" in domain:
            prof = result["metadata"].get("author_profile", {})
            print(f"[AUTHOR] Harvard Health: requests data ✅")
            print(f"[AUTHOR] Institution: {prof.get('institution', 'N/A')}")
            print(f"[AUTHOR] Credentials: {prof.get('credentials', [])}")
            print(f"[AUTHOR] Bio        : {prof.get('bio', '')[:60]}...")

        elif "medium.com" in domain:
            username = _extract_username(article_url)
            if not username:
                print(f"[AUTHOR] ⚠ Cannot extract username")
                continue

            author_data = _scrape_single_author(
                about_url   = f"https://medium.com/@{username}/about",
                profile_url = f"https://medium.com/@{username}",
                author_name = author_name,
            )
            result["metadata"]["author_profile"] = author_data

        else:
            print(f"[AUTHOR] Skipping: unsupported domain {domain}")

        time.sleep(settings.request_delay)

    return results

# ============================================================
# Public API
# ============================================================

def scrape_all_blogs(urls: list) -> list:
    """
    Full pipeline for list of blog URLs.

    Phase 1: Scrape articles (content, metadata, claps)
    Phase 2: Scrape author profiles
    Retry once per article on failure.
    """
    # ── Phase 1 ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Article Scraping ({len(urls)} URLs)")
    print(f"{'='*60}")

    results = []
    total   = len(urls)

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{total}] Scraping article...")

        result = _scrape_single_article(url)

        # Retry once if failed
        if (
            result["metadata"].get("error")
            or result["metadata"].get("word_count", 0) == 0
        ):
            print(f"\n[RETRY] First attempt failed → retrying...")
            time.sleep(3)
            result = _scrape_single_article(url)

            if result["metadata"].get("error"):
                print(f"[RETRY] ❌ Second attempt also failed")
            else:
                print(f"[RETRY] ✅ Succeeded on retry")

        results.append(result)

    # ── Phase 2 ────────────────────────────────────────────
    results = _scrape_all_author_profiles(results)

    return results


# ============================================================
# Direct Runner
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  Blog Scraper — Medium + Nature.com")
    print("=" * 60)
    print("\n  Supported URLs:")
    print("    → https://medium.com/@username/article-slug")
    print("    → https://www.nature.com/articles/article-id")
    print("\n  Enter 3 article URLs:\n")

    urls = []
    for i in range(3):
        while True:
            raw = input(f"  URL {i+1}: ").strip()
            if raw.startswith("http"):
                urls.append(raw)
                break
            print("  ⚠  Must start with http/https")

    print(f"\n▶ Starting pipeline...\n")

    results = scrape_all_blogs(urls)

    # ── Save ───────────────────────────────────────────────
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "blogs.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 85)
    print(f"  ✅ Saved → {output_path}")
    print("=" * 85)

    print(
        f"\n{'#':<3} {'Domain':<18} {'Author':<28} "
        f"{'Words':>7} {'Chunks':>7} {'Claps':>7} "
        f"{'Followers':>10} {'Bio':>4} Status"
    )
    print("-" * 90)

    for i, r in enumerate(results, 1):
        domain    = r["metadata"].get("domain", "?")[:17]
        author    = r.get("author", "?")[:27]
        words     = r["metadata"].get("word_count", 0)
        chunks    = r["metadata"].get("chunk_count", 0)
        claps     = r["metadata"].get("clap_count", 0)
        prof      = r["metadata"].get("author_profile", {})
        ok        = "✅" if chunks > 0 else "❌"
        err       = r["metadata"].get("error", "")

        if "nature.com" in domain:
            inst = prof.get("institution", "N/A")[:14]
            print(
                f"{i:<3} {domain:<18} {author:<28} "
                f"{words:>7} {chunks:>7} {'N/A':>7} "
                f"{inst:>10}  {'✅' if prof.get('bio') else '❌'}   {ok}"
            )
        else:
            followers = prof.get("followers", 0)
            has_bio   = "✅" if prof.get("bio") else "❌"
            print(
                f"{i:<3} {domain:<18} {author:<28} "
                f"{words:>7} {chunks:>7} {claps:>7,} "
                f"{followers:>10,}  {has_bio}   {ok}"
            )
        if err:
            print(f"    └─ Error: {err[:60]}")

    print()