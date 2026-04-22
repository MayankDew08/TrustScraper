# backend/scraper/blog_scraper.py
# ============================================================
# Playwright-based Medium blog scraper
#
# Two-phase pipeline:
#   Phase 1: Scrape articles (content, claps, metadata)
#   Phase 2: Scrape author profiles (followers, bio)
#
# Why two phases?
# → Article pages have media blocking for speed
# → Author about pages need clean browser (no blocking)
# → Separating concerns = more reliable scraping
# ============================================================

import json
import os
import sys
import time
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PWTimeout,
    Page,
    BrowserContext,
)
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.chunking import chunk_text
from utils.tagging import extract_tags
from utils.language_detector import detect_language


# ============================================================
# Shared Helpers
# ============================================================

def _parse_count_text(text: str) -> int:
    """
    Convert count strings to integers.
    "2.1K" → 2100, "499" → 499, "12K" → 12000
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
    Extract @username from Medium article URL.
    "https://medium.com/@Dr.Shlain/the-long-game..." → "Dr.Shlain"
    """
    try:
        match = re.search(r'medium\.com/@([^/]+)', article_url)
        if match:
            return match.group(1)
    except Exception:
        pass
    return ""


def _get_domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "").lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    """Scroll to trigger lazy loading."""
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


def _detect_medical_disclaimer(text: str) -> bool:
    phrases = [
        "consult a doctor", "consult your physician",
        "not a substitute", "healthcare professional",
        "seek medical attention", "talk to your doctor",
        "not medical advice", "for informational purposes only",
        "always consult", "speak with a qualified",
        "not intended to diagnose", "medical professional",
        "before starting any", "consult with a",
        "professional medical",
    ]
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def _clean_raw_text(raw_text: str) -> str:
    """Clean Playwright inner_text() — remove nav, footer, noise."""
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
# Metadata Extraction (BS4 from rendered HTML)
# ============================================================

def _extract_json_ld(soup: BeautifulSoup) -> dict:
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
    def try_parse(s) -> Optional[str]:
        if not s:
            return None
        try:
            return date_parser.parse(str(s), fuzzy=True).strftime("%Y-%m-%d")
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
# PHASE 1 — Article Scraping
# ============================================================

def _scrape_clap_count(page: Page) -> int:
    """Scrape clap count from article page."""
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


def _scrape_single_article(url: str) -> dict:
    """
    Phase 1: Scrape a single article.
    Opens its own browser → closes when done.
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

        # Block media for speed
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ["image", "media", "font"]
            else route.continue_()
        )

        try:
            # ── Load article ──────────────────────────────
            print(f"[PLAYWRIGHT] Loading article...")
            _navigate(page, url)
            _scroll_page(page)
            print(f"[PLAYWRIGHT] Page: {len(page.content()):,} bytes")

            # ── Claps ────────────────────────────────────
            clap_count = _scrape_clap_count(page)

            # ── Metadata ─────────────────────────────────
            html    = page.content()
            soup    = BeautifulSoup(html, "lxml")
            json_ld = _extract_json_ld(soup)
            title   = _extract_title(soup, json_ld)
            author  = _extract_author(soup, json_ld)
            pub_date= _extract_date(soup, json_ld)

            print(f"\n[META] Title  : {title[:70]}")
            print(f"[META] Author : {author}")
            print(f"[META] Date   : {pub_date}")
            print(f"[META] Claps  : {clap_count:,}")

            # ── Full text ────────────────────────────────
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

    # ── Clean + NLP ──────────────────────────────────────
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
# PHASE 2 — Author Profile Scraping
# ============================================================

def _find_followers_in_body(body_text: str) -> int:
    """
    Extract follower count from raw body text.
    Tested: "2.1K followers" → 2100
    """
    if not body_text:
        return 0

    lines = [l.strip() for l in body_text.split("\n") if l.strip()]

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # Same line: "2.1K followers"
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

        # Line is just "Followers"
        if line_lower in ("followers", "follower"):
            if i > 0:
                count = _parse_count_text(lines[i - 1].strip())
                if count > 0:
                    return count

    # Full text scan
    match = re.search(r'([\d]+\.?[\d]*\s*[KkMm]?)\s*[Ff]ollower', body_text)
    if match:
        count = _parse_count_text(match.group(1))
        if count > 0:
            print(f"[FOLLOWERS] Full-text: '{match.group(0)}' → {count:,}")
            return count

    return 0


def _find_bio_in_body(body_text: str, author_name: str) -> str:
    """
    Extract bio text from Medium About page.
    Looks for substantial text between follower section and footer.
    """
    lines = [l.strip() for l in body_text.split("\n") if l.strip()]

    skip = {
        "get app", "write", "sign up", "sign in",
        "home", "activity", "new", "about",
        "follow", "following", "·",
        "help", "status", "careers", "press", "blog",
        "privacy", "rules", "terms", "text to speech",
    }

    author_lower = author_name.lower()
    bio_lines    = []
    past_followers = False

    for line in lines:
        line_lower = line.lower().strip()

        # Mark when we pass the follower section
        if "follower" in line_lower:
            past_followers = True
            continue

        if not past_followers:
            continue

        # Skip noise
        if line_lower in skip:
            continue
        if line_lower == author_lower:
            continue
        if line_lower.startswith("connect with"):
            continue
        if "medium member since" in line_lower:
            continue
        if re.match(r'^[\d,.]+\s*[KkMm]?\s*(followers?|following)$', line, re.IGNORECASE):
            continue
        if re.match(r'^see all \(\d+\)$', line_lower):
            continue
        if len(line) < 15:
            continue

        # After followers section, short 2-3 word lines = followed user names
        if len(line.split()) <= 3 and len(line) < 40:
            continue

        bio_lines.append(line)
        if len(bio_lines) >= 3:
            break

    return " ".join(bio_lines)


def _find_article_count_in_body(body_text: str) -> int:
    match = re.search(
        r'(\d+)\s*(stories|story|posts|articles)',
        body_text,
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else 0


def _scrape_all_author_profiles(results: list) -> list:
    """
    Phase 2: Scrape author profiles.
    
    Fixed: Opens FRESH browser per author to avoid
    Medium rate limiting / session detection.
    Adds delay between requests.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Author Profile Scraping")
    print(f"{'='*60}")

    # Deduplicate authors
    username_map = {}
    for i, result in enumerate(results):
        url = result.get("source_url", "")
        username = _extract_username(url)
        if username:
            if username not in username_map:
                username_map[username] = []
            username_map[username].append(i)

    if not username_map:
        print(f"[AUTHOR] No usernames found")
        return results

    print(f"[AUTHOR] Found {len(username_map)} unique authors")

    for username, indices in username_map.items():
        about_url   = f"https://medium.com/@{username}/about"
        profile_url = f"https://medium.com/@{username}"
        author_name = results[indices[0]].get("author", "")

        print(f"\n[AUTHOR] @{username} → {about_url}")

        author_data = _scrape_single_author(about_url, profile_url, author_name)

        # Update all results by this author
        for idx in indices:
            results[idx]["metadata"]["author_profile"] = author_data

        # Delay between authors to avoid rate limiting
        time.sleep(3)

    return results


def _scrape_single_author(about_url: str, profile_url: str, author_name: str) -> dict:
    """
    Scrape a single author's about page.
    Opens FRESH browser → navigates → extracts → closes.
    
    Why fresh browser each time?
    → Medium tracks session across pages
    → After 1-2 requests in same context, returns login page
    → Fresh browser = fresh session = no rate limit
    
    Retries once on failure.
    """
    default = {
        "followers":     0,
        "bio":           "",
        "article_count": 0,
        "profile_url":   profile_url,
    }

    for attempt in range(2):  # Max 2 attempts
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
                    # Navigate to about page
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

                    # Get body text
                    body_text = page.inner_text("body")
                    print(f"[AUTHOR] Page text: {len(body_text)} chars")

                    # Check if we got real content or login page
                    if len(body_text) < 300:
                        print(f"[AUTHOR] ⚠ Page too short ({len(body_text)} chars) — likely blocked")
                        browser.close()

                        if attempt == 0:
                            continue  # Retry
                        else:
                            return default

                    # Extract data
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
                    else:
                        return default

        except Exception as e:
            print(f"[AUTHOR] ⚠ Browser error: {e}")

            if attempt == 0:
                continue
            else:
                return default

    return default

# ============================================================
# Public API
# ============================================================

def scrape_all_blogs(urls: list) -> list:
    """
    Full pipeline:
      Phase 1: Scrape all articles (content, claps, metadata)
      Phase 2: Scrape all author profiles (followers, bio)
      Retry once per article on failure.
    """
    # ── Phase 1: Articles ──────────────────────────────────
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

    # ── Phase 2: Author Profiles ───────────────────────────
    results = _scrape_all_author_profiles(results)

    return results


# ============================================================
# Direct Runner
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  Blog Scraper (Playwright) — Medium Articles")
    print("=" * 60)
    print("\n  Enter 3 Medium article URLs:\n")

    urls = []
    for i in range(3):
        while True:
            raw = input(f"  URL {i+1}: ").strip()
            if raw.startswith("http"):
                urls.append(raw)
                break
            print("  ⚠  Must start with http/https")

    print(f"\n▶ Starting scrape pipeline...\n")

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
    print("\n" + "=" * 75)
    print(f"  ✅ Saved → {output_path}")
    print("=" * 75)

    print(
        f"\n{'#':<3} {'Author':<30} {'Words':>7} "
        f"{'Chunks':>7} {'Tags':>5} {'Claps':>7} "
        f"{'Followers':>10} {'Bio':>4} Status"
    )
    print("-" * 85)

    for i, r in enumerate(results, 1):
        author    = r.get("author", "Unknown")[:29]
        words     = r["metadata"].get("word_count", 0)
        chunks    = r["metadata"].get("chunk_count", 0)
        tags      = len(r["topic_tags"])
        claps     = r["metadata"].get("clap_count", 0)
        prof      = r["metadata"].get("author_profile", {})
        followers = prof.get("followers", 0)
        has_bio   = "✅" if prof.get("bio") else "❌"
        ok        = "✅" if chunks > 0 else "❌"
        err       = r["metadata"].get("error", "")

        print(
            f"{i:<3} {author:<30} {words:>7} "
            f"{chunks:>7} {tags:>5} {claps:>7,} "
            f"{followers:>10,}  {has_bio}   {ok}"
        )
        if err:
            print(f"    └─ Error: {err[:55]}")

    print()