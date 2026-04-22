# backend/scraper/pubmed_scraper.py
# ============================================================
# PubMed Scraper — NCBI E-utilities API
#
# Why API instead of Playwright?
# → PubMed triggers reCAPTCHA on automated browsers
# → NCBI provides an official free REST API
# → More reliable, faster, structured data
# → Professional approach (what real researchers use)
#
# API Flow:
#   esearch → find PMIDs for query
#   efetch  → get article metadata (XML)
#   PMC API → get full text if available
# ============================================================

import json
import os
import sys
import time
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse, urlencode

import requests
from tenacity import retry, stop_after_attempt, wait_fixed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.chunking import chunk_text
from utils.tagging import extract_tags
from utils.language_detector import detect_language


# ============================================================
# NCBI API Constants
# ============================================================

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
PMC_URL     = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
PMC_OA_URL  = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"


# ============================================================
# HTTP
# ============================================================

def _get_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ResearchBot/1.0; "
            "mailto:research@example.com)"
        ),
        "Accept": "application/json, text/xml, */*",
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
)
def _api_get(url: str, params: dict) -> requests.Response:
    """Make GET request to NCBI API with retry."""
    response = requests.get(
        url,
        params=params,
        headers=_get_headers(),
        timeout=15,
    )
    response.raise_for_status()
    return response

def _extract_authors_from_xml(article) -> list:
    """Extract author names from any PubMed XML format."""
    authors = []
    for author in article.findall(".//Author"):
        last = author.findtext("LastName", "")
        fore = author.findtext("ForeName", "")
        coll = author.findtext("CollectiveName", "")

        if last:
            name = f"{fore} {last}".strip() if fore else last
            authors.append(name)
        elif coll:
            authors.append(coll)

    return authors


def _extract_abstract(article) -> str:
    """Extract abstract from any PubMed XML format."""
    abstract_parts = []
    for abstract_text in article.findall(".//AbstractText"):
        label = abstract_text.get("Label", "")
        text  = "".join(abstract_text.itertext()).strip()
        if text:
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

    return "\n".join(abstract_parts)


def _print_metadata(result: dict):
    """Pretty print metadata after extraction."""
    print(f"[EFETCH] ✅ Title    : {result['title'][:70]}")
    print(f"[EFETCH] ✅ Authors  : {result['author'][:70]}")
    print(f"[EFETCH] ✅ Journal  : {result['journal']}")
    print(f"[EFETCH] ✅ Date     : {result['pub_date']}")
    print(f"[EFETCH] ✅ PMC ID   : {result['pmc_id'] or 'Not available'}")
    print(f"[EFETCH] ✅ DOI      : {result['doi'] or 'Not available'}")
    print(f"[EFETCH] ✅ Citations: {result['citation_count']}")
# ============================================================
# Step 1 — Search: Get PMIDs
# ============================================================

def _search_pubmed(query: str, max_results: int = 5) -> list:
    """
    Search PubMed for journal articles matching query.
    Filters: journal articles only, English, recent.
    Returns list of PMIDs.
    """
    print(f"[ESEARCH] Searching PubMed for: '{query}'")

    # Build search term — filter for journal articles with abstracts
    search_term = f"({query}) AND journal article[pt] AND hasabstract[text]"

    params = {
        "db":       "pubmed",
        "term":     search_term,
        "retmode":  "json",
        "retmax":   max_results,
        "sort":     "relevance",
    }

    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key

    try:
        # First try: with free full text filter
        params_fft = {**params, "term": f"{search_term} AND free full text[filter]"}
        response = _api_get(ESEARCH_URL, params_fft)
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        total = data.get("esearchresult", {}).get("count", "0")

        print(f"[ESEARCH] Free full text results: {total}")

        if pmids:
            print(f"[ESEARCH] PMIDs (free full text): {pmids}")
            return pmids

        # Second try: without free full text filter
        print(f"[ESEARCH] No free full text → retrying without filter")
        response = _api_get(ESEARCH_URL, params)
        data = response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        total = data.get("esearchresult", {}).get("count", "0")

        print(f"[ESEARCH] Total results: {total}")
        print(f"[ESEARCH] PMIDs: {pmids}")

        return pmids

    except Exception as e:
        print(f"[ERROR] esearch failed: {e}")
        return []
# ============================================================
# Step 2 — Fetch: Get Article Metadata
# ============================================================

def _fetch_article_metadata(pmid: str) -> dict:
    """
    Fetch article metadata from PubMed via efetch API.
    Handles BOTH PubmedArticle and PubmedBookArticle XML formats.
    """
    print(f"[EFETCH] Fetching metadata for PMID: {pmid}")

    params = {
        "db":      "pubmed",
        "id":      pmid,
        "retmode": "xml",
        "rettype": "abstract",
    }

    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key

    try:
        response = _api_get(EFETCH_URL, params)
        root = ET.fromstring(response.content)

        # Try standard article first, then book article
        article = root.find(".//PubmedArticle")
        is_book = False

        if article is None:
            article = root.find(".//PubmedBookArticle")
            is_book = True

        if article is None:
            print(f"[ERROR] No article or book found for PMID: {pmid}")
            return {}

        if is_book:
            return _parse_book_article(article, pmid)
        else:
            return _parse_journal_article(article, pmid)

    except Exception as e:
        print(f"[ERROR] efetch failed for PMID {pmid}: {e}")
        return {}


def _parse_journal_article(article, pmid: str) -> dict:
    """
    Parse standard PubmedArticle XML.
    This is the most common type — journal research articles.
    """
    # ── Title ──────────────────────────────────────────
    title_el = article.find(".//ArticleTitle")
    title = (
        "".join(title_el.itertext()).strip()
        if title_el is not None
        else "Unknown Title"
    )

    # ── Authors ────────────────────────────────────────
    authors = _extract_authors_from_xml(article)
    author_str = ", ".join(authors) if authors else "Unknown Author"

    # ── Journal ────────────────────────────────────────
    journal_el = article.find(".//Journal/Title")
    journal = (
        journal_el.text.strip()
        if journal_el is not None and journal_el.text
        else "Unknown Journal"
    )

    # ── Date ───────────────────────────────────────────
    pub_date = _extract_pubmed_date(article)

    # ── Abstract ───────────────────────────────────────
    abstract = _extract_abstract(article)

    # ── DOI ────────────────────────────────────────────
    doi = ""
    for aid in article.findall(".//ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            doi = aid.text.strip()
            break
    # Also check ELocationID
    if not doi:
        eloc = article.find(".//ELocationID[@EIdType='doi']")
        if eloc is not None and eloc.text:
            doi = eloc.text.strip()

    # ── PMC ID ─────────────────────────────────────────
    pmc_id = ""
    for aid in article.findall(".//ArticleId"):
        if aid.get("IdType") == "pmc" and aid.text:
            pmc_id = aid.text.strip()
            break

    # ── Citation Count ─────────────────────────────────
    citation_count = _get_citation_count(pmid)

    result = {
        "pmid":           pmid,
        "title":          title,
        "author":         author_str,
        "authors_list":   authors,
        "journal":        journal,
        "pub_date":       pub_date,
        "abstract":       abstract,
        "doi":            doi,
        "pmc_id":         pmc_id,
        "citation_count": citation_count,
        "source_url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }

    _print_metadata(result)
    return result


def _parse_book_article(article, pmid: str) -> dict:
    """
    Parse PubmedBookArticle XML.
    Different structure from journal articles.
    """
    # ── Title ──────────────────────────────────────────
    title_el = (
        article.find(".//BookTitle")
        or article.find(".//ArticleTitle")
    )
    title = (
        "".join(title_el.itertext()).strip()
        if title_el is not None
        else "Unknown Title"
    )

    # ── Authors ────────────────────────────────────────
    authors = _extract_authors_from_xml(article)
    author_str = ", ".join(authors) if authors else "Unknown Author"

    # ── Publisher as Journal ───────────────────────────
    publisher = article.findtext(".//PublisherName", "Unknown Publisher")

    # ── Date ───────────────────────────────────────────
    pub_date = _extract_pubmed_date(article)

    # ── Abstract ───────────────────────────────────────
    abstract = _extract_abstract(article)

    # ── DOI ────────────────────────────────────────────
    doi = ""
    for aid in article.findall(".//ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            doi = aid.text.strip()
            break
    if not doi:
        eloc = article.find(".//ELocationID[@EIdType='doi']")
        if eloc is not None and eloc.text:
            doi = eloc.text.strip()

    # ── PMC ID ─────────────────────────────────────────
    pmc_id = ""
    book_acc = ""
    for aid in article.findall(".//ArticleId"):
        id_type = aid.get("IdType", "")
        if id_type == "pmc" and aid.text:
            pmc_id = aid.text.strip()
        elif id_type == "bookaccession" and aid.text:
            book_acc = aid.text.strip()

    citation_count = _get_citation_count(pmid)

    result = {
        "pmid":           pmid,
        "title":          title,
        "author":         author_str,
        "authors_list":   authors,
        "journal":        publisher,
        "pub_date":       pub_date,
        "abstract":       abstract,
        "doi":            doi,
        "pmc_id":         pmc_id or book_acc,
        "citation_count": citation_count,
        "source_url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }

    _print_metadata(result)
    return result

def _extract_pubmed_date(article) -> str:
    """
    Extract publication date from PubMed XML.
    Tries multiple date fields in order of reliability.
    """
    # Try PubDate first (most common)
    for date_path in [
        ".//PubDate",
        ".//ArticleDate",
        ".//PubMedPubDate[@PubStatus='pubmed']",
    ]:
        date_el = article.find(date_path)
        if date_el is not None:
            year  = date_el.findtext("Year",  "")
            month = date_el.findtext("Month", "01")
            day   = date_el.findtext("Day",   "01")

            if year:
                # Convert month name to number if needed
                month = _month_to_num(month)
                try:
                    return f"{year}-{int(month):02d}-{int(day):02d}"
                except ValueError:
                    return f"{year}-01-01"

    # Try MedlineDate (e.g., "2024 Jan-Feb")
    medline = article.findtext(".//MedlineDate", "")
    if medline:
        try:
            import re
            year_match = re.search(r'(19|20)\d{2}', medline)
            if year_match:
                return f"{year_match.group()}-01-01"
        except Exception:
            pass

    return "Unknown"


def _month_to_num(month: str) -> str:
    """Convert month name or number to zero-padded number string."""
    months = {
        "jan": "1", "feb": "2", "mar": "3", "apr": "4",
        "may": "5", "jun": "6", "jul": "7", "aug": "8",
        "sep": "9", "oct": "10","nov": "11","dec": "12",
    }
    if month.isdigit():
        return month
    return months.get(month.lower()[:3], "1")


# ============================================================
# Step 3 — Citation Count
# ============================================================

def _get_citation_count(pmid: str) -> int:
    """
    Get citation count for article using elink API.
    Uses PubMed Central citation database.

    Returns:
        Citation count (int), 0 if unavailable
    """
    try:
        params = {
            "dbfrom":  "pubmed",
            "db":      "pubmed",
            "id":      pmid,
            "linkname":"pubmed_pubmed_citedin",
            "retmode": "json",
        }
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key

        response = _api_get(ELINK_URL, params)
        data = response.json()

        links = (
            data.get("linksets", [{}])[0]
               .get("linksetdbs", [{}])[0]
               .get("links", [])
        )
        return len(links)

    except Exception:
        return 0


# ============================================================
# Step 4 — Full Text via PMC API
# ============================================================

def _fetch_full_text(pmc_id: str) -> Optional[str]:
    """
    Fetch full article text from PubMed Central OAI API.

    PMC OAI (Open Archives Initiative) provides free full text
    for open access articles in structured XML format.

    Args:
        pmc_id: PMC ID (e.g., "PMC1234567")

    Returns:
        Full text as string, or None if not available
    """
    if not pmc_id:
        print(f"[PMC] No PMC ID → full text not available")
        return None

    print(f"[PMC] Fetching full text for: {pmc_id}")

    # Ensure PMC ID has "PMC" prefix
    if not pmc_id.startswith("PMC"):
        pmc_id = f"PMC{pmc_id}"

    params = {
        "verb":             "GetRecord",
        "identifier":       f"oai:pubmedcentral.nih.gov:{pmc_id.replace('PMC', '')}",
        "metadataPrefix":   "pmc",
    }

    try:
        response = _api_get(PMC_OAI_URL, params)

        # Parse XML response
        root = ET.fromstring(response.content)

        # Define namespaces
        ns = {
            "oai": "http://www.openarchives.org/OAI/2.0/",
            "pmc": "https://jats.nlm.nih.gov/ns/archiving/1.0/",
        }

        # Extract all text from body sections
        body_texts = []

        # Try to find body/sec elements
        for elem in root.iter():
            # Get text from paragraph elements
            if elem.tag.endswith("}p") or elem.tag == "p":
                text = "".join(elem.itertext()).strip()
                if text and len(text) > 30:
                    body_texts.append(text)

        if body_texts:
            full_text = "\n".join(body_texts)
            print(f"[PMC] ✅ Full text: {len(full_text.split()):,} words")
            return full_text

        # Fallback: get all text from response
        all_text = " ".join(root.itertext()).strip()
        if len(all_text.split()) > 100:
            print(f"[PMC] ✅ Raw text: {len(all_text.split()):,} words")
            return all_text

        print(f"[PMC] ⚠ Empty response from PMC OAI")
        return None

    except Exception as e:
        print(f"[PMC] Full text fetch failed: {e}")
        return None


def _fetch_full_text_html(pmc_id: str) -> Optional[str]:
    """
    Fallback: fetch PMC article HTML page and extract text.
    Used when OAI API returns empty or fails.
    """
    if not pmc_id:
        return None

    if not pmc_id.startswith("PMC"):
        pmc_id = f"PMC{pmc_id}"

    url = PMC_URL.format(pmcid=pmc_id)
    print(f"[PMC] Trying HTML fallback: {url}")

    try:
        response = requests.get(url, headers=_get_headers(), timeout=15)
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "lxml")

        # PMC article body
        body = (
            soup.find("div", class_="jig-ncbiinpagenav-content")
            or soup.find("div", id="mc")
            or soup.find("article")
            or soup.find("main")
        )

        if not body:
            return None

        # Remove noise
        for tag in body.find_all(["script", "style", "nav", "aside"]):
            tag.decompose()

        # Extract paragraphs
        paragraphs = body.find_all("p")
        if paragraphs:
            texts = [
                p.get_text(separator=" ", strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 30
            ]
            if texts:
                full_text = "\n".join(texts)
                print(f"[PMC] ✅ HTML fallback: {len(full_text.split()):,} words")
                return full_text

        return None

    except Exception as e:
        print(f"[PMC] HTML fallback failed: {e}")
        return None


# ============================================================
# Text Cleaning
# ============================================================

def _clean_pubmed_text(text: str) -> str:
    """
    Clean PubMed/PMC text content.
    Removes references, copyright, figure legends.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned = []
    seen = set()
    skip_mode = False

    skip_headers = {
        "references", "bibliography", "acknowledgments",
        "acknowledgements", "conflict of interest",
        "funding", "author contributions",
        "supplementary material", "data availability",
    }

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Detect section headers to skip
        if stripped.lower() in skip_headers:
            skip_mode = True
            continue

        # Reset skip mode on new major section
        if skip_mode and len(stripped) > 100:
            skip_mode = False

        if skip_mode:
            continue

        # Skip short lines
        if len(stripped) < 30:
            continue

        # Skip duplicates
        if stripped in seen:
            continue

        # Skip reference entries (start with number)
        if re.match(r'^\d+\.?\s+[A-Z]', stripped):
            continue

        # Skip copyright lines
        lower = stripped.lower()
        if any(p in lower for p in [
            "copyright", "all rights reserved",
            "creative commons", "doi:", "published by",
            "©", "licensee",
        ]):
            continue

        # Skip low alpha ratio lines
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if alpha_ratio < 0.4:
            continue

        seen.add(stripped)
        cleaned.append(stripped)

    return "\n".join(cleaned)


# ============================================================
# Helpers
# ============================================================

def _detect_medical_disclaimer(text: str) -> bool:
    phrases = [
        "consult a doctor", "not a substitute",
        "healthcare professional", "seek medical",
        "not medical advice", "for informational purposes",
        "always consult", "not intended to diagnose",
        "clinical advice", "professional advice",
    ]
    lower = text.lower()
    return any(p in lower for p in phrases)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_result(query: str, error_msg: str) -> dict:
    return {
        "source_url":     ESEARCH_URL,
        "source_type":    "pubmed",
        "title":          "Unknown",
        "author":         "Unknown Author",
        "published_date": "Unknown",
        "language":       "unknown",
        "region":         "Unknown",
        "topic_tags":     [],
        "trust_score":    None,
        "content_chunks": [],
        "metadata": {
            "search_query": query,
            "error":        error_msg,
            "scraped_at":   _now_iso(),
        },
    }


# ============================================================
# Core Pipeline
# ============================================================

def scrape_pubmed(domain_query: str) -> dict:
    """
    Full PubMed scraping pipeline via NCBI E-utilities API.

    Steps:
      1. esearch  → get PMIDs for query
      2. efetch   → get article metadata (XML)
      3. elink    → get citation count
      4. PMC OAI  → get full text (if open access)
      5. NLP      → language, disclaimer, tags, chunks

    Args:
        domain_query: User search term (e.g., "sports psychology")

    Returns:
        Structured dict matching assignment JSON schema
    """
    print(f"\n{'='*60}")
    print(f"[PUBMED] Query: '{domain_query}'")
    print(f"{'='*60}")

    # ── Step 1: Search ─────────────────────────────────────
    print(f"\n[STEP 1] Searching PubMed...")
    pmids = _search_pubmed(domain_query, max_results=5)

    if not pmids:
        return _error_result(domain_query, "No results found")

    # Use first PMID
    pmid = pmids[0]
    print(f"[STEP 1] ✅ Using PMID: {pmid}")

    # Polite delay (NCBI rate limit)
    time.sleep(0.5)

    # ── Step 2: Fetch Metadata ─────────────────────────────
    print(f"\n[STEP 2] Fetching article metadata...")
    metadata = _fetch_article_metadata(pmid)

    if not metadata:
        return _error_result(domain_query, f"Failed to fetch PMID: {pmid}")

    print(f"[STEP 2] ✅ Metadata extracted")
    time.sleep(0.5)

    # ── Step 3: Fetch Full Text ────────────────────────────
    print(f"\n[STEP 3] Fetching full text...")
    pmc_id = metadata.get("pmc_id", "")

    full_text_raw = None
    has_full_text = False

    if pmc_id:
        # Try OAI API first
        full_text_raw = _fetch_full_text(pmc_id)

        # Fallback to HTML scraping
        if not full_text_raw or len(full_text_raw.split()) < 100:
            print(f"[STEP 3] OAI returned little → trying HTML fallback")
            full_text_raw = _fetch_full_text_html(pmc_id)

        if full_text_raw and len(full_text_raw.split()) > 100:
            has_full_text = True
            content = _clean_pubmed_text(full_text_raw)
            print(f"[STEP 3] ✅ Full text: {len(content.split()):,} words")
        else:
            print(f"[STEP 3] ⚠ Full text unavailable → using abstract")
            content = metadata.get("abstract", "")
    else:
        print(f"[STEP 3] ⚠ No PMC ID → using abstract only")
        content = metadata.get("abstract", "")

    word_count = len(content.split())
    print(f"\n[CONTENT] Words   : {word_count:,}")
    print(f"[CONTENT] Preview : {content[:300]}...")

    # ── Step 4: NLP Pipeline ──────────────────────────────
    print(f"\n[NLP] Processing...")

    language = detect_language(content)
    print(f"[NLP] Language  : {language}")

    has_disclaimer = _detect_medical_disclaimer(content)
    print(f"[NLP] Disclaimer: {has_disclaimer}")

    print(f"[NLP] Extracting tags...")
    tags = extract_tags(content)
    print(f"[NLP] Tags      : {tags}")

    print(f"[NLP] Chunking...")
    chunks = chunk_text(content)
    print(f"[NLP] Chunks    : {len(chunks)}")

    # ── Build Result ───────────────────────────────────────
    return {
        "source_url":     metadata["source_url"],
        "source_type":    "pubmed",
        "title":          metadata["title"],
        "author":         metadata["author"],
        "published_date": metadata["pub_date"],
        "language":       language,
        "region":         "Unknown",
        "topic_tags":     tags,
        "trust_score":    None,
        "content_chunks": chunks,
        "metadata": {
            "domain":                 "pubmed.ncbi.nlm.nih.gov",
            "pmid":                   pmid,
            "pmc_id":                 pmc_id,
            "doi":                    metadata.get("doi", ""),
            "journal":                metadata["journal"],
            "citation_count":         metadata.get("citation_count", 0),
            "abstract":               metadata.get("abstract", "")[:500],
            "has_full_text":          has_full_text,
            "has_medical_disclaimer": has_disclaimer,
            "search_query":           domain_query,
            "word_count":             word_count,
            "chunk_count":            len(chunks),
            "scraped_at":             _now_iso(),
        },
    }


# ============================================================
# Direct Runner
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  PubMed Scraper — NCBI E-utilities API")
    print("=" * 60)

    domain = input(
        "\n  Enter search domain\n"
        "  (e.g., sports psychology, diabetes AI, nutrition):\n\n"
        "  → "
    ).strip()

    if not domain:
        domain = "machine learning healthcare"
        print(f"  Using default: '{domain}'")

    print(f"\n▶ Starting PubMed scraper for: '{domain}'\n")

    result = scrape_pubmed(domain)

    # Save
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pubmed.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([result], f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "=" * 60)
    print(f"  ✅ Saved → {output_path}")
    print("=" * 60)

    words  = result["metadata"].get("word_count", 0)
    chunks = result["metadata"].get("chunk_count", 0)
    tags   = len(result["topic_tags"])
    ok     = "✅" if chunks > 0 else "❌"
    ft     = result["metadata"].get("has_full_text", False)
    cit    = result["metadata"].get("citation_count", 0)

    print(f"\n  Title     : {result['title'][:65]}")
    print(f"  Authors   : {result['author'][:65]}")
    print(f"  Journal   : {result['metadata'].get('journal', '?')}")
    print(f"  PMID      : {result['metadata'].get('pmid', '?')}")
    print(f"  Citations : {cit}")
    print(f"  Full Text : {'Yes' if ft else 'Abstract only'}")
    print(f"  Words     : {words:,}")
    print(f"  Chunks    : {chunks}")
    print(f"  Tags      : {tags} → {result['topic_tags']}")
    print(f"  Status    : {ok}")
    print()