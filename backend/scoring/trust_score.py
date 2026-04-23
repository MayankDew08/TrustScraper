# backend/scoring/trust_score.py
# ============================================================
# Trust Score Engine
#
# Formula:
#   Trust Score = (
#       w1 × author_credibility   [0.30] +
#       w2 × citation_count       [0.20] +
#       w3 × domain_authority     [0.20] +
#       w4 × recency              [0.15] +
#       w5 × medical_disclaimer   [0.15]
#   ) × abuse_multiplier
#
# Score range: 0.0 – 1.0
#
# Source tiers:
#   institutional → Harvard, Nature, NIH, WHO (.edu/.gov)
#   open_platform → Medium, YouTube, unknown blogs
#   pubmed        → NCBI peer-reviewed database
# ============================================================

import math
import re
import sys
import os
import json
from datetime import datetime, timezone
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

# Import AI explainer — integrated into pipeline
from scoring.ai_explainer import generate_trust_explanation


# ============================================================
# Weights — must sum to 1.0
# ============================================================

WEIGHTS = {
    "author_credibility":  0.30,
    "citation_count":      0.20,
    "domain_authority":    0.20,
    "recency":             0.15,
    "medical_disclaimer":  0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ============================================================
# Credential Keywords
# ============================================================

NAME_CREDENTIALS = [
    "md", "m.d", "m.d.",
    "phd", "ph.d", "ph.d.",
    "do", "d.o", "d.o.",
    "dds", "dmd",
    "rn", "np", "pa", "pa-c",
    "psyd", "psy.d",
    "msw", "lcsw", "mft",
    "mph", "drph",
    "bcpa",
    "mba", "msc", "m.sc",
    "prof", "prof.",
    "dr", "dr.",
]

BIO_CREDENTIALS = [
    "professor", "researcher", "scientist",
    "specialist", "consultant", "director",
    "expert", "physician", "surgeon",
    "psychiatrist", "psychologist", "therapist",
    "nutritionist", "dietitian", "clinician",
    "practitioner", "faculty", "lecturer",
    "board certified", "board-certified",
    "fellowship", "residency", "attending",
    "medical director", "chief", "founder",
    "published", "peer-reviewed", "peer reviewed",
]

INSTITUTION_SIGNALS = [
    "university", "institute", "hospital", "college",
    "school of", "department of", "research center",
    "foundation", "laboratory", "clinic", "center",
    "mit", "stanford", "harvard", "oxford",
    "cambridge", "yale", "johns hopkins",
    "nih", "who", "cdc", "fda", "mayo",
    "kaiser", "cleveland clinic",
]


# ============================================================
# Domain Authority Maps
# ============================================================

DOMAIN_AUTHORITY = {
    # Academic / Government — Tier 1
    "pubmed.ncbi.nlm.nih.gov":  1.00,
    "ncbi.nlm.nih.gov":         1.00,
    "nih.gov":                  1.00,
    "who.int":                  1.00,
    "cdc.gov":                  1.00,
    "nature.com":               1.00,
    "science.org":              1.00,
    "nejm.org":                 1.00,
    "thelancet.com":            1.00,
    "jamanetwork.com":          0.97,
    "bmj.com":                  0.97,

    # Reputable Medical Institutions — Tier 2
    "health.harvard.edu":       0.93,
    "mayoclinic.org":           0.92,
    "hopkinsmedicine.org":      0.90,
    "clevelandclinic.org":      0.88,
    "medlineplus.gov":          0.88,
    "webmd.com":                0.80,
    "healthline.com":           0.80,
    "medicalnewstoday.com":     0.78,
    "harvard.edu":              0.95,
    "stanford.edu":             0.95,
    "ox.ac.uk":                 0.93,
    "cam.ac.uk":                0.92,

    # Quality Tech / AI — Tier 3
    "realpython.com":           0.75,
    "machinelearningmastery.com": 0.72,
    "towardsdatascience.com":   0.70,

    # Open Platforms — Penalized Base + Author Ceiling
    "medium.com":               0.45,
    "youtube.com":              0.52,
    "substack.com":             0.42,
    "wordpress.com":            0.35,
    "blogger.com":              0.30,
}

# Institutional domains (get floor scores + implied trust)
INSTITUTIONAL_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    "ncbi.nlm.nih.gov":        0.95,
    "nih.gov":                 0.92,
    "who.int":                 0.92,
    "cdc.gov":                 0.92,
    "nature.com":              0.90,
    "science.org":             0.90,
    "nejm.org":                0.90,
    "thelancet.com":           0.90,
    "jamanetwork.com":         0.88,
    "bmj.com":                 0.88,
    "health.harvard.edu":      0.88,
    "mayoclinic.org":          0.87,
    "hopkinsmedicine.org":     0.86,
    "clevelandclinic.org":     0.85,
    "medlineplus.gov":         0.85,
    "harvard.edu":             0.88,
    "stanford.edu":            0.88,
    "ox.ac.uk":                0.86,
    "cam.ac.uk":               0.86,
}

# Open platform ceilings — max score regardless of author
MEDIUM_CEILING  = 0.82
YOUTUBE_CEILING = 0.78

# Medium publications with editorial standards
MEDIUM_PUBLICATIONS = {
    "towards data science":        0.72,
    "towardsdatascience":          0.72,
    "the startup":                 0.62,
    "thestartup":                  0.62,
    "better humans":               0.62,
    "entrepreneurship handbook":   0.62,
    "mind cafe":                   0.60,
    "tincture":                    0.65,
    "elemental":                   0.65,
    "personal growth":             0.57,
    "lifestyle":                   0.55,
    "careers":                     0.57,
    "productivity":                0.60,
}

# Medical disclaimer phrases
DISCLAIMER_PHRASES = [
    "consult a doctor", "consult your physician",
    "not a substitute", "healthcare professional",
    "seek medical attention", "talk to your doctor",
    "not medical advice", "for informational purposes only",
    "always consult", "speak with a qualified",
    "not intended to diagnose", "medical professional",
    "before starting any", "consult with a",
    "professional medical advice",
]

# Medical topic signals
MEDICAL_TOPIC_SIGNALS = [
    "health", "medical", "disease", "treatment",
    "diagnosis", "drug", "medication", "therapy",
    "clinical", "symptom", "nutrition", "diet",
    "mental health", "depression", "anxiety",
    "cancer", "diabetes", "obesity", "vaccine",
    "supplement", "exercise", "fitness", "wellness",
    "nervous system", "brain", "psychology",
    "neuroscience", "cognitive", "behavior",
    "doctor", "hospital", "patient", "cure",
]

PERSONAL_ESSAY_SIGNALS = [
    "in my experience", "i remember", "i learned",
    "my story", "personal journey", "when i was",
    "i decided", "i felt", "i noticed",
    "from my perspective", "i believe", "in my opinion",
]

MEDICAL_ADVICE_SIGNALS = [
    "you should take", "you should use", "recommended dose",
    "take this medication", "treatment for", "cure for",
    "remedy for", "how to treat", "helps treat",
    "proven to cure", "clinically proven",
    "supplement", "dosage", "side effects",
]

MISINFORMATION_PHRASES = [
    "doctors don't want you to know",
    "big pharma doesn't want",
    "miracle cure", "100% natural cure",
    "guaranteed to cure", "secret remedy",
    "they don't want you to know",
    "proven to cure cancer",
    "cures all diseases", "miracle drug",
    "no side effects", "ancient secret",
]


# ============================================================
# Source Tier Classification
# ============================================================

def _get_source_tier(domain: str) -> str:
    """
    Classify domain into trust tier.

    Returns:
        "institutional" → Harvard, Nature, NIH, WHO, .edu, .gov
        "open"          → Medium, YouTube, unknown blogs
    """
    domain = domain.lower().replace("www.", "")

    if domain in INSTITUTIONAL_DOMAINS:
        return "institutional"
    if domain.endswith(".edu") or domain.endswith(".gov"):
        return "institutional"
    if ".ac." in domain:
        return "institutional"

    return "open"


# ============================================================
# Citation Count in Content
# ============================================================

def _count_citations_in_content(content: str) -> int:
    """
    Count academic citations in article text.

    Detects:
    - Author (Year): Smith (2022), Smith et al. (2022)
    - DOI mentions
    - Journal names
    """
    if not content:
        return 0

    patterns = [
        r'\b[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\(\d{4}\)',
        r'\b(?:doi|DOI):\s*10\.',
        r'\b(?:Journal|Proceedings|Review|Lancet|Nature|Science|NEJM)\b',
    ]

    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, content))

    return min(count, 10)


# ============================================================
# Variable 1 — Author Credibility
# ============================================================

def _score_pubmed_author(author: str) -> Tuple[float, dict]:
    """
    PubMed authors get elevated floor scores.
    Author count = peer review signal.
    """
    if not author or author == "Unknown Author":
        return 0.50, {"reason": "no author but pubmed source"}

    count = len([a for a in author.split(",") if a.strip()])

    if count >= 5:   score = 0.95
    elif count >= 3: score = 0.90
    elif count >= 2: score = 0.85
    else:            score = 0.80

    return score, {
        "author_count": count,
        "source":       "pubmed",
        "tier":         "pubmed",
    }


def _score_institutional_author(
    author: str,
    bio: str,
    domain: str,
    author_profile: dict,
    content: str,
) -> Tuple[float, dict]:
    """
    Score author from institutional source (Harvard, Nature, NIH).

    Starts from institution floor score.
    Boosted by credentials, affiliation, ORCID.
    NOT penalized for missing followers/claps.
    """
    floor_score = INSTITUTIONAL_DOMAINS.get(domain, 0.80)

    if domain.endswith(".edu"):
        floor_score = max(floor_score, 0.85)
    if domain.endswith(".gov"):
        floor_score = max(floor_score, 0.88)

    score = floor_score

    signals = {
        "author_name":       author,
        "institution_floor": floor_score,
        "tier":              "institutional",
        "credentials_found": [],
        "has_bio":           bool(bio),
    }

    author_lower = (author or "").lower()
    bio_lower    = (bio or "").lower()

    # Credentials in author name
    name_creds = []
    for cred in NAME_CREDENTIALS:
        pattern = r'\b' + re.escape(cred) + r'\b'
        if re.search(pattern, author_lower):
            name_creds.append(cred)
    if name_creds:
        cred_boost = min(len(name_creds) * 0.03, 0.08)
        score += cred_boost
        signals["credentials_found"].extend(name_creds)
        print(f"[CREDIBILITY] Name creds {name_creds} → +{cred_boost:.3f}")

    # Credentials in bio
    bio_creds = [c for c in BIO_CREDENTIALS if c in bio_lower]
    if bio_creds:
        bio_boost = min(len(bio_creds) * 0.02, 0.05)
        score += bio_boost
        signals["credentials_found"].extend(bio_creds[:3])
        print(f"[CREDIBILITY] Bio creds {bio_creds[:3]} → +{bio_boost:.3f}")

    # Profile credentials (e.g., Harvard author page)
    profile_creds = author_profile.get("credentials", [])
    if profile_creds:
        profile_boost = min(len(profile_creds) * 0.02, 0.05)
        score += profile_boost
        signals["credentials_found"].extend(profile_creds)
        print(f"[CREDIBILITY] Profile creds {profile_creds} → +{profile_boost:.3f}")

    # Institution in bio
    if any(inst in bio_lower for inst in INSTITUTION_SIGNALS):
        score += 0.02
        print(f"[CREDIBILITY] Institution in bio → +0.02")

    # ORCID verified
    if author_profile.get("orcid"):
        score += 0.02
        print(f"[CREDIBILITY] ORCID verified → +0.02")

    # Citations in content
    citation_count = _count_citations_in_content(content)
    if citation_count >= 5:
        score += 0.03
        print(f"[CREDIBILITY] {citation_count} citations → +0.03")
    elif citation_count >= 2:
        score += 0.02
    signals["citations_in_content"] = citation_count

    final = round(min(1.0, score), 3)
    print(f"[CREDIBILITY] Institutional final: {final} (floor={floor_score})")
    return final, signals


def _score_open_platform_author(
    author: str,
    bio: str,
    followers: int,
    article_count: int,
    content: str,
) -> Tuple[float, dict]:
    """
    Score author from open platform (Medium, blogs).

    Starts from 0.25 base.
    Built up by credentials, institution, followers, citations.
    Penalized for generic/anonymous name.
    """
    if not author or author.strip() in ("", "Unknown Author"):
        return 0.10, {"reason": "no author"}

    author_lower = author.lower().strip()
    bio_lower    = (bio or "").lower().strip()
    score        = 0.25

    signals = {
        "author_name":       author,
        "tier":              "open_platform",
        "credentials_found": [],
        "has_bio":           bool(bio),
        "followers":         followers,
        "article_count":     article_count,
    }

    # Credentials in name
    name_creds = []
    for cred in NAME_CREDENTIALS:
        pattern = r'\b' + re.escape(cred) + r'\b'
        if re.search(pattern, author_lower):
            name_creds.append(cred)
    if name_creds:
        cred_boost = min(len(name_creds) * 0.18, 0.40)
        score += cred_boost
        signals["credentials_found"].extend(name_creds)
        print(f"[CREDIBILITY] Name creds {name_creds} → +{cred_boost:.3f}")

    # Credentials in bio
    bio_creds = [c for c in BIO_CREDENTIALS if c in bio_lower]
    if bio_creds:
        bio_boost = min(len(bio_creds) * 0.08, 0.20)
        score += bio_boost
        signals["credentials_found"].extend(bio_creds[:3])
        print(f"[CREDIBILITY] Bio creds {bio_creds[:3]} → +{bio_boost:.3f}")

    # Institution in bio
    if any(inst in bio_lower for inst in INSTITUTION_SIGNALS):
        score += 0.20
        print(f"[CREDIBILITY] Institution in bio → +0.20")

    # Real name pattern
    clean_name = re.sub(
        r'\b(' + '|'.join(re.escape(c) for c in NAME_CREDENTIALS) + r')\b',
        '', author_lower
    ).strip(" ,.")
    parts = clean_name.split()
    is_real_name = (
        len(parts) >= 2
        and not any(c.isdigit() for c in clean_name)
        and all(c.isalpha() or c in " '-.,/" for c in clean_name)
    )
    if is_real_name:
        score += 0.08
        print(f"[CREDIBILITY] Real name → +0.08")

    # Follower count (log-scaled, max +0.15)
    if followers > 0:
        follower_boost = min(math.log10(followers + 1) / 25.0, 0.15)
        score += follower_boost
        signals["follower_boost"] = round(follower_boost, 3)
        print(f"[CREDIBILITY] Followers {followers:,} → +{follower_boost:.3f}")

    # Article count
    if article_count >= 50:   score += 0.05
    elif article_count >= 20: score += 0.03
    elif article_count >= 5:  score += 0.01

    # Citations in content
    citation_count = _count_citations_in_content(content)
    if citation_count >= 5:
        score += 0.12
        print(f"[CREDIBILITY] {citation_count} citations → +0.12")
    elif citation_count >= 2:
        score += 0.07
    elif citation_count >= 1:
        score += 0.03
    signals["citations_in_content"] = citation_count

    # Generic name penalty
    GENERIC = [
        "admin", "staff", "editor", "team", "anonymous",
        "user", "author", "writer", "unknown", "guest",
    ]
    if any(g in author_lower for g in GENERIC):
        score -= 0.20
        print(f"[CREDIBILITY] Generic name penalty → -0.20")

    final = round(max(0.0, min(1.0, score)), 3)
    print(f"[CREDIBILITY] Open platform final: {final}")
    return final, signals


def _score_youtube_author(
    channel_name: str,
    subscriber_count: int,
    channel_description: str,
    total_videos: int,
    view_count: int,
    like_count: int,
    content: str,
) -> Tuple[float, dict]:
    """
    Score YouTube channel credibility.

    Signals:
    1. Subscriber count (log-scaled)
    2. Like/view ratio (quality proxy)
    3. Credentials in channel name/description
    4. Institution mentions
    5. Total videos (experience)
    6. Citations in content
    """
    score   = 0.20
    signals = {
        "channel_name":      channel_name,
        "subscriber_count":  subscriber_count,
        "view_count":        view_count,
        "like_count":        like_count,
        "credentials_found": [],
        "tier":              "open_platform",
    }

    name_lower = (channel_name or "").lower()
    desc_lower = (channel_description or "").lower()
    combined   = f"{name_lower} {desc_lower}"

    # Subscriber count (log-scaled, max +0.25)
    if subscriber_count > 0:
        sub_boost = min(math.log10(subscriber_count + 1) / 28.0, 0.25)
        score += sub_boost
        signals["subscriber_boost"] = round(sub_boost, 4)
        print(f"[YT-CRED] Subscribers {subscriber_count:,} → +{sub_boost:.4f}")

    # Like/view ratio (max +0.15)
    if view_count > 0 and like_count > 0:
        like_ratio  = like_count / view_count
        ratio_score = min(like_ratio / 0.08, 1.0) * 0.15
        score += ratio_score
        signals["like_view_ratio"]  = round(like_ratio, 4)
        signals["like_ratio_boost"] = round(ratio_score, 4)
        print(f"[YT-CRED] Like ratio {like_ratio:.2%} → +{ratio_score:.4f}")

    # Credentials in name/description
    name_creds = []
    for cred in NAME_CREDENTIALS:
        pattern = r'\b' + re.escape(cred) + r'\b'
        if re.search(pattern, combined):
            name_creds.append(cred)
    if name_creds:
        cred_boost = min(len(name_creds) * 0.10, 0.20)
        score += cred_boost
        signals["credentials_found"].extend(name_creds)
        print(f"[YT-CRED] Credentials {name_creds} → +{cred_boost:.4f}")

    # Institution in description
    if any(inst in desc_lower for inst in INSTITUTION_SIGNALS):
        score += 0.10
        signals["institution_found"] = True
        print(f"[YT-CRED] Institution found → +0.10")

    # Total videos (experience)
    if total_videos >= 500:   score += 0.05
    elif total_videos >= 100: score += 0.03
    elif total_videos >= 10:  score += 0.01

    # Citations in content
    citation_count = _count_citations_in_content(content)
    if citation_count >= 5:
        score += 0.08
        print(f"[YT-CRED] {citation_count} citations → +0.08")
    elif citation_count >= 2:
        score += 0.04
    signals["citations_in_content"] = citation_count

    final = round(max(0.0, min(1.0, score)), 3)
    print(f"[YT-CRED] Final: {final}")
    return final, signals


def _score_multiple_authors(
    author_str: str,
    bio: str,
    source_type: str,
    domain: str,
    author_profile: dict,
    content: str,
) -> Tuple[float, dict]:
    """
    Handle multiple authors — average credibility scores.

    Assignment requirement:
    "Multiple Authors → Use average credibility score"

    Also applies collaboration bonus when 2+ authors
    have credentials (peer review signal).
    """
    CREDENTIAL_SUFFIXES = {
        "md", "phd", "do", "rn", "np", "pa", "dds",
        "mph", "psyd", "lcsw", "msw", "mba", "msc",
        "bcpa", "facc", "facs",
    }

    # Smart split — handle "Marc B. Garnick, MD"
    raw_parts = [p.strip() for p in author_str.split(",")]
    authors   = []
    i = 0
    while i < len(raw_parts):
        part = raw_parts[i].strip()
        if (
            i + 1 < len(raw_parts)
            and raw_parts[i + 1].strip().lower().rstrip(".") in CREDENTIAL_SUFFIXES
        ):
            part = f"{part} {raw_parts[i + 1].strip()}"
            i += 2
        else:
            i += 1
        if part and len(part) > 2:
            authors.append(part)

    if not authors:
        authors = [author_str]

    print(f"[MULTI-AUTHOR] {len(authors)} authors: {authors}")

    individual_scores = []
    all_credentials   = []
    tier              = _get_source_tier(domain)

    for author in authors:
        if source_type == "pubmed":
            score, sigs = _score_pubmed_author(author)
        elif tier == "institutional":
            score, sigs = _score_institutional_author(
                author         = author,
                bio            = bio,
                domain         = domain,
                author_profile = author_profile,
                content        = content,
            )
        else:
            score, sigs = _score_open_platform_author(
                author        = author,
                bio           = bio,
                followers     = author_profile.get("followers", 0),
                article_count = author_profile.get("article_count", 0),
                content       = content,
            )

        individual_scores.append(score)
        all_credentials.extend(sigs.get("credentials_found", []))
        print(f"[MULTI-AUTHOR]   {author[:40]} → {score:.3f}")

    # Average score
    avg_score = round(sum(individual_scores) / len(individual_scores), 3)

    # Collaboration bonus (2+ credentialed authors)
    credentialed_count = sum(1 for s in individual_scores if s >= 0.60)
    if credentialed_count >= 2:
        collab_bonus = min(credentialed_count * 0.02, 0.06)
        avg_score    = round(min(1.0, avg_score + collab_bonus), 3)
        print(
            f"[MULTI-AUTHOR] Collaboration bonus +{collab_bonus:.2f} "
            f"({credentialed_count} credentialed authors)"
        )

    print(
        f"[MULTI-AUTHOR] Scores: {[round(s,3) for s in individual_scores]} "
        f"→ Average: {avg_score:.3f}"
    )

    signals = {
        "author_name":          author_str,
        "author_count":         len(authors),
        "individual_scores":    individual_scores,
        "average_score":        avg_score,
        "credentials_found":    list(dict.fromkeys(all_credentials)),
        "credentialed_authors": credentialed_count,
        "tier":                 tier,
    }

    return avg_score, signals


def score_author_credibility(
    author: str,
    bio: str = "",
    followers: int = 0,
    article_count: int = 0,
    source_type: str = "blog",
    content: str = "",
    domain: str = "",
    author_profile: dict = None,
) -> Tuple[float, dict]:
    """
    Score author credibility.

    Routes to:
    - _score_multiple_authors()    if comma-separated names
    - _score_pubmed_author()       if source_type == pubmed
    - _score_institutional_author() if institutional domain
    - _score_open_platform_author() otherwise
    """
    author_profile = author_profile or {}
    domain         = domain.lower().replace("www.", "")

    if not author or author.strip() in ("", "Unknown Author"):
        return 0.10, {"reason": "no author", "tier": "unknown"}

    # ── Detect multiple authors ────────────────────────────
    parts = [p.strip() for p in author.split(",") if p.strip()]
    CRED_SUFFIXES = {
        "md", "phd", "do", "rn", "np", "pa", "dds",
        "mph", "psyd", "lcsw", "msw", "mba", "msc",
        "bcpa", "facc", "facs",
    }
    has_multiple = (
        len(parts) >= 2
        and not all(
            p.lower().rstrip(".") in CRED_SUFFIXES
            for p in parts[1:]
        )
        and len([
            p for p in parts
            if p.lower().rstrip(".") not in CRED_SUFFIXES
            and len(p) > 3
        ]) >= 2
    )

    if has_multiple:
        print(f"[AUTHOR] Multiple authors: '{author[:60]}'")
        return _score_multiple_authors(
            author_str     = author,
            bio            = bio,
            source_type    = source_type,
            domain         = domain,
            author_profile = author_profile,
            content        = content,
        )

    # ── PubMed ─────────────────────────────────────────────
    if source_type == "pubmed":
        return _score_pubmed_author(author)

    # ── Institutional ──────────────────────────────────────
    tier = _get_source_tier(domain)
    if tier == "institutional":
        return _score_institutional_author(
            author         = author,
            bio            = bio,
            domain         = domain,
            author_profile = author_profile,
            content        = content,
        )

    # ── Open Platform ──────────────────────────────────────
    return _score_open_platform_author(
        author        = author,
        bio           = bio,
        followers     = followers,
        article_count = article_count,
        content       = content,
    )


# ============================================================
# Variable 2 — Citation / Engagement Score
# ============================================================

def score_citation_count(
    citation_count: int = 0,
    clap_count: int = 0,
    source_type: str = "blog",
    domain: str = "",
    view_count: int = 0,
    like_count: int = 0,
) -> float:
    """
    Score citations (PubMed) or engagement (blogs/YouTube).

    PubMed:      log10(citations + 1) / 2.0
    Blog:        log10(claps + 1) / 4.0
    YouTube:     combined view + like score
    Institutional: neutral 0.60 (editorial quality implied)
    """
    domain = domain.lower().replace("www.", "")
    tier   = _get_source_tier(domain)

    # PubMed — real citations
    if source_type == "pubmed":
        if citation_count == 0:
            return 0.20
        score = min(math.log10(citation_count + 1) / 2.0, 1.0)
        return round(score, 3)

    # Institutional — editorial process implies quality
    if tier == "institutional":
        print(f"[CITATION] Institutional → neutral 0.60")
        return 0.60

    # YouTube — view count + like count combined
    if source_type == "youtube":
        view_score = 0.0
        like_score = 0.0

        if view_count > 0:
            view_score = min(math.log10(view_count + 1) / 8.0, 1.0)
        if like_count > 0:
            like_score = min(math.log10(like_count + 1) / 6.0, 1.0)

        combined = (view_score * 0.6) + (like_score * 0.4)
        result   = round(min(combined, 1.0), 3)

        print(
            f"[CITATION] YouTube: "
            f"views={view_count:,}({view_score:.3f}) "
            f"likes={like_count:,}({like_score:.3f}) "
            f"→ {result:.3f}"
        )
        return result

    # Blog — clap count as engagement proxy
    if clap_count == 0:
        return 0.10
    score = min(math.log10(clap_count + 1) / 4.0, 1.0)
    return round(score, 3)


# ============================================================
# Variable 3 — Domain Authority
# ============================================================

def score_domain_authority(
    domain: str,
    content: str = "",
    source_type: str = "blog",
    author_score: float = 0.0,
) -> float:
    """
    Score domain authority.

    Institutional: hardcoded tier score
    Medium:        base 0.45 + author boost, ceiling 0.82
    YouTube:       base 0.52 + channel boost, ceiling 0.78
    Unknown:       0.38 base
    """
    domain = domain.lower().replace("www.", "")

    # PubMed / NCBI
    if "ncbi.nlm.nih.gov" in domain or "pubmed" in domain:
        return 1.00

    # Academic domains
    if domain.endswith(".edu"):
        return 0.88
    if ".ac." in domain:
        return 0.85
    if domain.endswith(".gov"):
        return 0.90

    base_score = DOMAIN_AUTHORITY.get(domain, 0.38)

    # Medium — boosted by author, capped at ceiling
    if domain == "medium.com":
        boost = 0.0
        if author_score >= 0.85:
            boost += 0.20
            print(f"[DOMAIN] Medium author boost +0.20")
        elif author_score >= 0.70:
            boost += 0.12
            print(f"[DOMAIN] Medium author boost +0.12")
        elif author_score >= 0.55:
            boost += 0.06
            print(f"[DOMAIN] Medium author boost +0.06")

        # Known publication boost
        if content:
            pub_boost = _detect_medium_publication(content)
            if pub_boost > 0:
                boost += pub_boost

        raw   = base_score + boost
        final = round(min(raw, MEDIUM_CEILING), 3)
        print(
            f"[DOMAIN] Medium: base={base_score} "
            f"+ boost={boost:.2f} = {raw:.3f} "
            f"→ capped at {final}"
        )
        return final

    # YouTube — boosted by channel, capped at ceiling
    if domain == "youtube.com":
        boost = 0.0
        if author_score >= 0.85:   boost += 0.18
        elif author_score >= 0.70: boost += 0.10
        elif author_score >= 0.55: boost += 0.05

        raw   = base_score + boost
        final = round(min(raw, YOUTUBE_CEILING), 3)
        print(
            f"[DOMAIN] YouTube: base={base_score} "
            f"+ boost={boost:.2f} = {raw:.3f} "
            f"→ capped at {final}"
        )
        return final

    # Substack / WordPress / Blogger
    if domain in ("substack.com", "wordpress.com", "blogger.com"):
        if author_score >= 0.80:   boost = 0.15
        elif author_score >= 0.60: boost = 0.08
        else:                       boost = 0.0
        return round(min(base_score + boost, 0.70), 3)

    return round(base_score, 3)


def _detect_medium_publication(content: str) -> float:
    """Detect known Medium publication → return authority boost."""
    content_lower = content.lower()
    best_boost    = 0.0
    for pub_name, pub_score in MEDIUM_PUBLICATIONS.items():
        if pub_name in content_lower:
            boost = pub_score - 0.45
            if boost > best_boost:
                best_boost = boost
                print(f"[DOMAIN] Publication '{pub_name}' → boost +{boost:.2f}")
    return round(best_boost, 3)


# ============================================================
# Variable 4 — Recency
# ============================================================

def score_recency(published_date: str) -> float:
    """
    Score content recency by age from today.

    Thresholds:
      0-90 days    → 1.00
      91-180 days  → 0.90
      181-365 days → 0.80
      1-2 years    → 0.60
      2-3 years    → 0.40
      3-5 years    → 0.25
      5+ years     → 0.10
      Unknown      → 0.40 (mild penalty)
    """
    if not published_date or published_date in ("Unknown", ""):
        return 0.40

    try:
        pub   = datetime.strptime(published_date, "%Y-%m-%d")
        today = datetime.now()
        age   = (today - pub).days
    except Exception:
        return 0.40

    if age < 0:      return 0.50   # Future date = parsing error
    if age <= 90:    return 1.00
    if age <= 180:   return 0.90
    if age <= 365:   return 0.80
    if age <= 730:   return 0.60
    if age <= 1095:  return 0.40
    if age <= 1825:  return 0.25
    return 0.10


# ============================================================
# Variable 5 — Medical Disclaimer
# ============================================================

def score_medical_disclaimer(
    content: str,
    topic_tags: list,
    has_disclaimer: bool = False,
    domain: str = "",
) -> Tuple[float, bool]:
    """
    Score medical disclaimer presence.

    Institutional sources:
      → Implied disclaimer 0.85 (editorial standards)
      → Explicit disclaimer 1.00

    Open platforms:
      → Non-medical content → 0.50 (neutral)
      → Medical + disclaimer → 1.00
      → Medical + direct advice + no disclaimer → 0.00
      → Medical + personal essay → 0.40
      → Medical + no disclaimer → 0.20

    Returns:
        (score: float, is_medical: bool)
    """
    domain = domain.lower().replace("www.", "")
    tier   = _get_source_tier(domain)

    if tier == "institutional":
        content_lower = content.lower()
        has_explicit  = has_disclaimer or any(
            p in content_lower for p in DISCLAIMER_PHRASES
        )
        score = 1.0 if has_explicit else 0.85
        print(
            f"[DISCLAIMER] Institutional → "
            f"{'explicit' if has_explicit else 'implied'}: {score}"
        )
        return score, True

    # Open platform
    content_lower = content.lower()
    tags_lower    = " ".join(topic_tags).lower()
    combined      = f"{content_lower} {tags_lower}"

    has_explicit = has_disclaimer or any(
        p in content_lower for p in DISCLAIMER_PHRASES
    )

    medical_count = sum(
        1 for s in MEDICAL_TOPIC_SIGNALS if s in combined
    )
    is_medical = medical_count >= 2

    if not is_medical:
        return 0.50, False

    if has_explicit:
        return 1.00, True

    personal_count = sum(
        1 for s in PERSONAL_ESSAY_SIGNALS if s in content_lower
    )
    is_personal_essay = personal_count >= 3

    advice_count = sum(
        1 for s in MEDICAL_ADVICE_SIGNALS if s in content_lower
    )
    is_medical_advice = advice_count >= 2

    if is_medical_advice:
        return 0.00, True
    if is_personal_essay:
        return 0.40, True

    return 0.20, True


# ============================================================
# Abuse Detection
# ============================================================

def compute_abuse_multiplier(
    author: str,
    content: str,
    domain: str,
    topic_tags: list,
    source_type: str = "blog",
    metadata: dict = None,
) -> Tuple[float, list]:
    """
    Detect manipulation / spam signals.

    Checks:
      1. Fake authors (gibberish, bots, unverifiable claims)
      2. SEO spam (keyword stuffing, thin content, bad domain)
      3. Misleading medical content (claims without disclaimer)
      4. Outdated info (handled by recency score, flagged here)
      5. YouTube no transcript (content quality signal)

    Returns:
        (multiplier: float [0.3-1.0], issues: list[str])
    """
    multiplier = 1.0
    issues     = []
    metadata   = metadata or {}

    # ── Check 1: Fake Authors ─────────────────────────────
    if author and author not in ("Unknown Author", ""):
        author_lower = author.lower()

        # Gibberish name (low vowel ratio)
        letters = [c for c in author_lower if c.isalpha()]
        if letters:
            vowel_ratio = sum(
                1 for c in letters if c in "aeiou"
            ) / len(letters)
            if vowel_ratio < 0.10:
                multiplier -= 0.25
                issues.append(
                    f"FAKE AUTHOR: Gibberish name "
                    f"(vowel ratio {vowel_ratio:.2f})"
                )

        # Numbers in name
        if re.search(r'\d{3,}', author):
            multiplier -= 0.20
            issues.append("FAKE AUTHOR: Suspicious numbers in name")

        # Too short
        if len(author.strip()) < 3:
            multiplier -= 0.20
            issues.append("FAKE AUTHOR: Name too short")

        # All caps
        if author.isupper() and len(author) > 4:
            multiplier -= 0.10
            issues.append("FAKE AUTHOR: All caps name")

        # Bot patterns
        bot_patterns = [
            r'^user\d+$', r'^admin\d*$', r'^test\d*$',
            r'^guest\d*$', r'^bot\d*$',
        ]
        for pattern in bot_patterns:
            if re.match(pattern, author_lower):
                multiplier -= 0.30
                issues.append(
                    f"FAKE AUTHOR: Matches bot pattern '{pattern}'"
                )
                break

        # Credential claimed but no bio (cannot verify)
        name_has_cred = any(
            re.search(r'\b' + re.escape(c) + r'\b', author_lower)
            for c in NAME_CREDENTIALS
        )
        author_profile = metadata.get("author_profile", {})
        has_bio        = bool(author_profile.get("bio", ""))
        if name_has_cred and not has_bio:
            issues.append(
                "AUTHOR NOTE: Credentials in name but no bio found "
                "— cannot fully verify"
            )

    # ── Check 2: SEO Spam ─────────────────────────────────
    if content:
        words       = content.lower().split()
        total_words = len(words)

        if total_words > 0:
            # Keyword stuffing
            word_freq = {}
            for w in words:
                if len(w) > 4:
                    word_freq[w] = word_freq.get(w, 0) + 1

            for word, count in word_freq.items():
                if count / total_words > 0.05:
                    multiplier -= 0.15
                    issues.append(
                        f"SEO SPAM: Keyword stuffing — "
                        f"'{word}' = {count/total_words:.1%} of content"
                    )
                    break

        # Thin content on low-authority domain
        domain_clean = domain.lower().replace("www.", "")
        da = DOMAIN_AUTHORITY.get(domain_clean, 0.38)
        if total_words < 200 and da < 0.60:
            multiplier -= 0.15
            issues.append(
                f"SEO SPAM: Thin content ({total_words} words) "
                f"on low-authority domain"
            )

    # Spam domain patterns
    domain_lower = domain.lower()
    SPAM_PATTERNS = [r'\d{4,}', r'free.*money', r'click.*here', r'best.*review']
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, domain_lower):
            multiplier -= 0.15
            issues.append(f"SEO SPAM: Suspicious domain pattern")
            break

    # YouTube description spam
    desc_analysis = metadata.get("description_analysis", {})
    if desc_analysis.get("spam_signals", 0) >= 3:
        multiplier -= 0.15
        issues.append(
            f"SEO SPAM: {desc_analysis['spam_signals']} "
            f"promotional phrases in description"
        )

    # ── Check 3: Misleading Medical Content ──────────────
    content_lower = (content or "").lower()

    # Misinformation phrases
    for phrase in MISINFORMATION_PHRASES:
        if phrase in content_lower:
            multiplier -= 0.30
            issues.append(
                f"MISLEADING MEDICAL: '{phrase}'"
            )
            break

    # Medical advice + no disclaimer
    has_disclaimer = metadata.get("has_medical_disclaimer", False)
    has_advice     = any(
        s in content_lower for s in MEDICAL_ADVICE_SIGNALS
    )
    if has_advice and not has_disclaimer:
        multiplier -= 0.20
        issues.append(
            "MISLEADING MEDICAL: Direct medical advice without disclaimer"
        )

    # ── Check 4: Outdated Info ────────────────────────────
    # Handled by recency score (0.10 for 5+ year old content)
    # No additional multiplier to avoid double-penalty

    # ── Check 5: YouTube No Transcript ───────────────────
    if source_type == "youtube":
        transcript_source = metadata.get("transcript_source", "none")
        has_transcript    = metadata.get("has_transcript", False)
        view_count        = metadata.get("view_count", 0)

        if transcript_source == "none" and not has_transcript:
            multiplier -= 0.08
            issues.append(
                "NO TRANSCRIPT: Content analysis based on "
                "description only — reduced confidence"
            )

            # Extra penalty for popular video with no captions
            if view_count > 500_000:
                multiplier -= 0.05
                issues.append(
                    f"NO TRANSCRIPT: Popular video ({view_count:,} views) "
                    f"with no captions — possibly intentional"
                )

    return max(0.30, round(multiplier, 2)), issues


# ============================================================
# Main Trust Score Function
# ============================================================

def compute_trust_score(article: dict) -> dict:
    """
    Compute trust score for a single scraped article.

    Full pipeline:
      1. Extract all signals from article dict
      2. Route to correct author scorer by source type
      3. Score all 5 variables
      4. Apply weighted sum
      5. Apply abuse multiplier
      6. Generate AI explanation (Groq + prompt.txt)
      7. Return complete breakdown dict

    Args:
        article: Scraped article dict from any scraper

    Returns:
        Complete breakdown dict including ai_explanation
    """
    source_type = article.get("source_type", "blog")
    metadata    = article.get("metadata", {})
    content     = " ".join(article.get("content_chunks", []))
    topic_tags  = article.get("topic_tags", [])
    domain      = metadata.get("domain", "")

    author         = article.get("author", "Unknown Author")
    pub_date       = article.get("published_date", "Unknown")
    author_profile = metadata.get("author_profile", {})
    bio            = author_profile.get("bio", "")
    followers      = author_profile.get("followers", 0)
    article_count  = author_profile.get("article_count", 0)
    clap_count     = metadata.get("clap_count", 0)
    citation_count = metadata.get("citation_count", 0)
    has_disclaimer = metadata.get("has_medical_disclaimer", False)

    # YouTube-specific signals
    view_count       = metadata.get("view_count",          0)
    like_count       = metadata.get("like_count",          0)
    subscriber_count = metadata.get("subscriber_count",    0)
    channel_desc     = metadata.get("channel_description", "")
    total_videos     = metadata.get("channel_total_videos",0)

    print(f"\n{'='*55}")
    print(f"[TRUST] {article.get('title', '')[:50]}")
    print(f"[TRUST] Source: {source_type} | Domain: {domain}")
    print(f"[TRUST] Tier  : {_get_source_tier(domain)}")
    print(f"{'='*55}")

    # ── Variable 1: Author Credibility ────────────────────
    if source_type == "youtube":
        v1_author, author_signals = _score_youtube_author(
            channel_name        = author,
            subscriber_count    = subscriber_count,
            channel_description = channel_desc,
            total_videos        = total_videos,
            view_count          = view_count,
            like_count          = like_count,
            content             = content,
        )
    else:
        v1_author, author_signals = score_author_credibility(
            author         = author,
            bio            = bio,
            followers      = followers,
            article_count  = article_count,
            source_type    = source_type,
            content        = content,
            domain         = domain,
            author_profile = author_profile,
        )

    # ── Variable 2: Citation / Engagement ─────────────────
    v2_citation = score_citation_count(
        citation_count = citation_count,
        clap_count     = clap_count,
        source_type    = source_type,
        domain         = domain,
        view_count     = view_count,
        like_count     = like_count,
    )

    # ── Variable 3: Domain Authority ──────────────────────
    v3_domain = score_domain_authority(
        domain       = domain,
        content      = content,
        source_type  = source_type,
        author_score = v1_author,
    )

    # ── Variable 4: Recency ───────────────────────────────
    v4_recency = score_recency(pub_date)

    # ── Variable 5: Medical Disclaimer ───────────────────
    v5_disclaimer, is_medical = score_medical_disclaimer(
        content        = content,
        topic_tags     = topic_tags,
        has_disclaimer = has_disclaimer,
        domain         = domain,
    )

    # ── Weighted Sum ──────────────────────────────────────
    weighted_sum = (
        WEIGHTS["author_credibility"] * v1_author    +
        WEIGHTS["citation_count"]     * v2_citation  +
        WEIGHTS["domain_authority"]   * v3_domain    +
        WEIGHTS["recency"]            * v4_recency   +
        WEIGHTS["medical_disclaimer"] * v5_disclaimer
    )

    # ── Abuse Multiplier ──────────────────────────────────
    abuse_multiplier, abuse_issues = compute_abuse_multiplier(
        author      = author,
        content     = content,
        domain      = domain,
        topic_tags  = topic_tags,
        source_type = source_type,
        metadata    = metadata,
    )

    # ── Final Score ───────────────────────────────────────
    final_score = round(
        max(0.0, min(1.0, weighted_sum * abuse_multiplier)), 3
    )

    # ── Print Breakdown ───────────────────────────────────
    print(f"\n[TRUST] Score Breakdown:")
    print(f"  author_credibility : {v1_author:.3f} × {WEIGHTS['author_credibility']} = {v1_author * WEIGHTS['author_credibility']:.4f}")
    print(f"  citation_count     : {v2_citation:.3f} × {WEIGHTS['citation_count']} = {v2_citation * WEIGHTS['citation_count']:.4f}")
    print(f"  domain_authority   : {v3_domain:.3f} × {WEIGHTS['domain_authority']} = {v3_domain * WEIGHTS['domain_authority']:.4f}")
    print(f"  recency            : {v4_recency:.3f} × {WEIGHTS['recency']} = {v4_recency * WEIGHTS['recency']:.4f}")
    print(f"  medical_disclaimer : {v5_disclaimer:.3f} × {WEIGHTS['medical_disclaimer']} = {v5_disclaimer * WEIGHTS['medical_disclaimer']:.4f}")
    print(f"  {'─'*48}")
    print(f"  Weighted Sum       : {weighted_sum:.4f}")
    print(f"  Abuse Multiplier   : × {abuse_multiplier}")
    for issue in abuse_issues:
        print(f"    ⚠ {issue}")
    print(f"  {'─'*48}")
    print(f"  FINAL SCORE        : {final_score:.3f}  ", end="")
    if final_score >= 0.80:   print("🟢 High Trust")
    elif final_score >= 0.60: print("🟡 Moderate")
    elif final_score >= 0.40: print("🟠 Low")
    else:                     print("🔴 Unreliable")

    # ── Build Breakdown Dict ──────────────────────────────
    breakdown = {
        "author_credibility": {
            "score":        v1_author,
            "weight":       WEIGHTS["author_credibility"],
            "contribution": round(v1_author * WEIGHTS["author_credibility"], 4),
            "signals":      author_signals,
        },
        "citation_count": {
            "score":        v2_citation,
            "weight":       WEIGHTS["citation_count"],
            "contribution": round(v2_citation * WEIGHTS["citation_count"], 4),
            "signals": {
                "clap_count":      clap_count,
                "citation_count":  citation_count,
                "view_count":      view_count,
                "like_count":      like_count,
                "domain_tier":     _get_source_tier(domain),
            },
        },
        "domain_authority": {
            "score":        v3_domain,
            "weight":       WEIGHTS["domain_authority"],
            "contribution": round(v3_domain * WEIGHTS["domain_authority"], 4),
            "signals":      {"domain": domain},
        },
        "recency": {
            "score":        v4_recency,
            "weight":       WEIGHTS["recency"],
            "contribution": round(v4_recency * WEIGHTS["recency"], 4),
            "signals":      {"published_date": pub_date},
        },
        "medical_disclaimer": {
            "score":        v5_disclaimer,
            "weight":       WEIGHTS["medical_disclaimer"],
            "contribution": round(v5_disclaimer * WEIGHTS["medical_disclaimer"], 4),
            "signals": {
                "has_disclaimer": has_disclaimer,
                "is_medical":     is_medical,
                "domain_tier":    _get_source_tier(domain),
            },
        },
        "abuse_detection": {
            "multiplier": abuse_multiplier,
            "issues":     abuse_issues,
        },
        "weighted_sum":  round(weighted_sum, 4),
        "final_score":   final_score,
    }

    # ── AI Explanation ────────────────────────────────────
    print(f"\n[AI EXPLAIN] Generating explanation...")
    try:
        ai_explanation = generate_trust_explanation(
            title       = article.get("title", "Unknown"),
            author      = author,
            domain      = domain,
            source_type = source_type,
            score       = final_score,
            breakdown   = breakdown,
        )
    except Exception as e:
        print(f"[AI EXPLAIN] ⚠ Failed: {e}")
        ai_explanation = {
            "summary":                f"Score: {final_score:.3f}",
            "mathematical_breakdown": {},
            "key_drivers":            [],
            "improvement_suggestions":[],
            "anomaly_flag":           False,
            "anomaly_reason":         None,
            "verification_questions": [],
        }

    breakdown["ai_explanation"] = ai_explanation

    return breakdown


# ============================================================
# Batch Scorer
# ============================================================

def score_all(articles: list) -> list:
    """
    Score all articles in a list.
    Attaches trust_score, trust_score_breakdown,
    and ai_explanation to each article dict.

    Prints anomaly warnings for flagged articles.
    """
    total = len(articles)

    for i, article in enumerate(articles, 1):
        print(
            f"\n[{i}/{total}] Scoring: "
            f"{article.get('title', 'Unknown')[:50]}"
        )

        breakdown = compute_trust_score(article)

        article["trust_score"]           = breakdown["final_score"]
        article["trust_score_breakdown"] = breakdown
        article["ai_explanation"]        = breakdown.get("ai_explanation", {})

        # Print anomaly warnings
        explanation = article["ai_explanation"]
        if explanation.get("anomaly_flag"):
            print(f"\n{'⚠' * 20}")
            print(f"  ANOMALY FLAGGED")
            print(f"  Article : {article.get('title', '')[:60]}")
            print(f"  Score   : {article['trust_score']:.3f}")
            print(f"  Domain  : {article.get('metadata', {}).get('domain', '')}")
            print(f"  Reason  : {explanation.get('anomaly_reason', '')}")
            print(f"  Verify  :")
            for q in explanation.get("verification_questions", [])[:3]:
                print(f"    → {q}")
            print(f"{'⚠' * 20}\n")

    return articles


# ============================================================
# Direct Runner — Score Existing JSON Files
# ============================================================

if __name__ == "__main__":

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )

    files = {
        "blogs.json":   os.path.join(output_dir, "blogs.json"),
        "pubmed.json":  os.path.join(output_dir, "pubmed.json"),
        "youtube.json": os.path.join(output_dir, "youtube.json"),
    }

    all_results = []

    for fname, fpath in files.items():
        if not os.path.exists(fpath):
            print(f"[SKIP] {fname} not found")
            continue

        print(f"\n{'='*55}")
        print(f"  Scoring: {fname}")
        print(f"{'='*55}")

        with open(fpath, encoding="utf-8") as f:
            articles = json.load(f)

        scored = score_all(articles)
        all_results.extend(scored)

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2, ensure_ascii=False)

        print(f"\n[SAVED] {fname} updated with trust scores")

    # ── Final Summary ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRUST SCORE SUMMARY")
    print(f"{'='*70}")
    print(
        f"{'Title':<42} {'Type':<8} {'Score':>6}  "
        f"{'Grade':<14} Flagged"
    )
    print(f"{'-'*75}")

    flagged = []
    for a in all_results:
        title  = (a.get("title") or "Unknown")[:41]
        stype  = a.get("source_type", "?")[:7]
        score  = a.get("trust_score", 0) or 0
        expl   = a.get("ai_explanation", {})
        flag   = expl.get("anomaly_flag", False)

        if score >= 0.80:   grade = "🟢 High Trust"
        elif score >= 0.60: grade = "🟡 Moderate"
        elif score >= 0.40: grade = "🟠 Low"
        else:               grade = "🔴 Unreliable"

        flag_str = "⚠ VERIFY" if flag else "✅ Clean"
        print(
            f"{title:<42} {stype:<8} {score:>6.3f}  "
            f"{grade:<14} {flag_str}"
        )

        if flag:
            flagged.append(a)

    if flagged:
        print(f"\n{'='*70}")
        print(f"  ⚠ ARTICLES REQUIRING VERIFICATION ({len(flagged)})")
        print(f"{'='*70}")
        for a in flagged:
            expl = a.get("ai_explanation", {})
            print(f"\n  {a.get('title','')[:65]}")
            print(f"  Score  : {a.get('trust_score', 0):.3f}")
            print(f"  Reason : {expl.get('anomaly_reason','')[:80]}")
            for q in expl.get("verification_questions", [])[:2]:
                print(f"    → {q}")

    scores = [a.get("trust_score", 0) or 0 for a in all_results]
    if scores:
        print(f"\n{'─'*50}")
        print(f"  Total    : {len(scores)}")
        print(f"  Average  : {sum(scores)/len(scores):.3f}")
        print(f"  Highest  : {max(scores):.3f}")
        print(f"  Lowest   : {min(scores):.3f}")
    print()