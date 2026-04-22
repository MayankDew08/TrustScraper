# backend/scoring/trust_score.py
# ============================================================
# Trust Score Engine — Updated for Medium blogs
#
# Key improvements:
#   1. Credential detection from author NAME (not just bio)
#   2. Follower count has meaningful impact
#   3. Domain authority boosted by author credentials
#   4. Medical disclaimer: distinguishes essays vs advice
#   5. Citation detection from article content
#   6. Platform credibility: proven author = no ceiling
# ============================================================

import math
import re
import sys
import os
from datetime import datetime, timezone
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


# ============================================================
# Weights — must sum to 1.0
# ============================================================

WEIGHTS = {
    "author_credibility":  0.30,  # Increased — most important signal
    "citation_count":      0.20,
    "domain_authority":    0.20,  # Reduced — platform matters less than author
    "recency":             0.15,
    "medical_disclaimer":  0.15,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ============================================================
# Credential Keywords
# ============================================================

# In author NAME (e.g., "Jordan L. Shlain MD")
NAME_CREDENTIALS = [
    "md", "m.d", "m.d.",
    "phd", "ph.d", "ph.d.",
    "do", "d.o", "d.o.",
    "dds", "dmd",
    "rn", "np", "pa", "pa-c",
    "psyd", "psy.d",
    "msw", "lcsw", "mft",
    "mph", "drph",
    "bcpa",                     # Board Certified Patient Advocate
    "mba", "msc", "m.sc",
    "prof", "prof.",
    "dr", "dr.",
]

# In bio text
BIO_CREDENTIALS = [
    "professor", "researcher", "scientist",
    "specialist", "consultant", "director",
    "expert", "physician", "surgeon",
    "psychiatrist", "psychologist", "therapist",
    "nutritionist", "dietitian", "clinician",
    "practitioner", "practising", "practicing",
    "faculty", "lecturer", "instructor",
    "board certified", "board-certified",
    "fellowship", "residency", "attending",
    "medical director", "chief", "founder",
    "published", "peer-reviewed", "peer reviewed",
]

# Institution signals in bio
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
# Domain Authority Map
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

    # Reputable Health — Tier 2
    "mayoclinic.org":           0.92,
    "webmd.com":                0.80,
    "healthline.com":           0.80,
    "medicalnewstoday.com":     0.78,
    "harvard.edu":              0.95,
    "stanford.edu":             0.95,
    "ox.ac.uk":                 0.93,

    # Quality Tech/AI — Tier 3
    "realpython.com":           0.75,
    "machinelearningmastery.com": 0.72,
    "towardsdatascience.com":   0.70,
    "youtube.com":              0.65,

    # Medium base — boosted by author credibility
    "medium.com":               0.55,
}

# Medium publications that have editorial standards
MEDIUM_PUBLICATIONS = {
    "towards data science":        0.75,
    "towardsdatascience":          0.75,
    "the startup":                 0.65,
    "thestartup":                  0.65,
    "better humans":               0.65,
    "entrepreneurship handbook":   0.65,
    "mind cafe":                   0.62,
    "tincture":                    0.68,  # Medical/health publication
    "elemental":                   0.68,  # Medium health publication
    "personal growth":             0.60,
    "lifestyle":                   0.58,
    "careers":                     0.60,
    "productivity":                0.62,
}


# ============================================================
# Variable 1 — Author Credibility
# ============================================================

def score_author_credibility(
    author: str,
    bio: str = "",
    followers: int = 0,
    article_count: int = 0,
    source_type: str = "blog",
    content: str = "",
) -> tuple:
    """
    Score author credibility.

    Key fix: Check credentials in AUTHOR NAME first.
    "Jordan L. Shlain MD" → "md" detected → high score.
    "Malynnda Stewart, PhD, BCPA" → "phd", "bcpa" detected.

    Also checks:
    - Bio text for expertise signals
    - Follower count (meaningful but not dominant)
    - Article count (experience signal)
    - Citations in content (research awareness)

    Returns:
        (score: float, signals_dict: dict)
    """
    if source_type == "pubmed":
        score, signals = _score_pubmed_author(author)
        return score, signals

    if not author or author.strip() in ("", "Unknown Author"):
        return 0.10, {"reason": "no author"}

    author_lower = author.lower().strip()
    bio_lower    = bio.lower().strip() if bio else ""
    combined     = f"{author_lower} {bio_lower}"

    score   = 0.25  # Base: named author exists
    signals = {
        "author_name":       author,
        "credentials_found": [],
        "has_bio":           bool(bio),
        "followers":         followers,
        "article_count":     article_count,
        "institution_found": False,
        "citations_in_content": 0,
    }

    # ── Signal 1: Credentials in AUTHOR NAME (strongest) ──
    # Check name specifically — not combined with bio
    # so we know credential is claimed in their identity
    name_creds_found = []
    for cred in NAME_CREDENTIALS:
        # Match as word boundary to avoid partial matches
        pattern = r'\b' + re.escape(cred) + r'\b'
        if re.search(pattern, author_lower):
            name_creds_found.append(cred)

    if name_creds_found:
        # Each unique credential adds to score
        cred_boost = min(len(name_creds_found) * 0.18, 0.40)
        score += cred_boost
        signals["credentials_found"].extend(name_creds_found)
        print(f"[CREDIBILITY] Name credentials: {name_creds_found} → +{cred_boost:.2f}")

    # ── Signal 2: Credentials in Bio ──────────────────────
    bio_creds_found = []
    for cred in BIO_CREDENTIALS:
        if cred in bio_lower:
            bio_creds_found.append(cred)

    if bio_creds_found:
        bio_boost = min(len(bio_creds_found) * 0.08, 0.20)
        score += bio_boost
        signals["credentials_found"].extend(bio_creds_found[:3])
        print(f"[CREDIBILITY] Bio credentials: {bio_creds_found[:3]} → +{bio_boost:.2f}")

    # ── Signal 3: Institution in Bio ──────────────────────
    institution_found = any(inst in bio_lower for inst in INSTITUTION_SIGNALS)
    if institution_found:
        score += 0.15
        signals["institution_found"] = True
        print(f"[CREDIBILITY] Institution found in bio → +0.15")

    # ── Signal 4: Real name pattern ───────────────────────
    # Strip credentials from name first for pattern check
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
        print(f"[CREDIBILITY] Real name pattern → +0.08")

    # ── Signal 5: Follower count ──────────────────────────
    # Log scale — meaningful but capped at 0.15
    # 500 followers   → +0.04
    # 2000 followers  → +0.08
    # 10000 followers → +0.12
    # 100000 followers → +0.15
    if followers > 0:
        follower_boost = min(
            math.log10(followers + 1) / 25.0,
            0.15
        )
        score += follower_boost
        signals["follower_boost"] = round(follower_boost, 3)
        print(f"[CREDIBILITY] Followers {followers:,} → +{follower_boost:.3f}")

    # ── Signal 6: Article count ───────────────────────────
    if article_count >= 50:
        score += 0.05
    elif article_count >= 20:
        score += 0.03
    elif article_count >= 5:
        score += 0.01

    # ── Signal 7: Citations in content ───────────────────
    # If author cites peer-reviewed papers → high credibility
    citation_count = _count_citations_in_content(content)
    if citation_count >= 5:
        score += 0.12
        print(f"[CREDIBILITY] {citation_count} citations found → +0.12")
    elif citation_count >= 2:
        score += 0.07
        print(f"[CREDIBILITY] {citation_count} citations found → +0.07")
    elif citation_count >= 1:
        score += 0.03

    signals["citations_in_content"] = citation_count

    # ── Penalty: Generic/anonymous name ──────────────────
    GENERIC = [
        "admin", "staff", "editor", "team", "anonymous",
        "user", "author", "writer", "unknown", "guest",
    ]
    if any(g in author_lower for g in GENERIC):
        score -= 0.20

    final = round(max(0.0, min(1.0, score)), 3)
    print(f"[CREDIBILITY] Final: {final}")
    return final, signals


def _count_citations_in_content(content: str) -> int:
    """
    Count academic citations in article content.

    Detects:
    - Author (Year) format: "Smith (2022)"
    - Journal references: "Journal of Psychology"
    - DOI mentions: "doi:"
    - Reference list patterns

    High citation count = author understands academic literature.
    """
    if not content:
        return 0

    patterns = [
        r'\b[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\(\d{4}\)',  # Smith (2022) or Smith et al. (2022)
        r'\b(?:doi|DOI):\s*10\.',                           # DOI mentions
        r'\b(?:Journal|Proceedings|Review|Lancet|Nature|Science|NEJM)\b',  # Journal names
    ]

    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, content)
        count += len(matches)

    # Cap at 10 to avoid inflation
    return min(count, 10)


def _score_pubmed_author(author: str) -> tuple:
    """PubMed authors get elevated floor scores."""
    if not author or author == "Unknown Author":
        return 0.50, {"reason": "no author but pubmed source"}

    count = len([a for a in author.split(",") if a.strip()])

    if count >= 5:
        score = 0.95
    elif count >= 3:
        score = 0.90
    elif count >= 2:
        score = 0.85
    else:
        score = 0.80

    return score, {"author_count": count, "source": "pubmed"}


# ============================================================
# Variable 2 — Citation / Engagement Score
# ============================================================

def score_citation_count(
    citation_count: int = 0,
    clap_count: int = 0,
    source_type: str = "blog",
) -> float:
    """
    Score citations (PubMed) or engagement (blogs).
    """
    if source_type == "pubmed":
        if citation_count == 0:
            return 0.20
        return round(min(math.log10(citation_count + 1) / 2.0, 1.0), 3)

    elif source_type == "blog":
        if clap_count == 0:
            return 0.10
        # log10(10000) = 4 → 1.0
        return round(min(math.log10(clap_count + 1) / 4.0, 1.0), 3)

    elif source_type == "youtube":
        return 0.50

    return 0.20


# ============================================================
# Variable 3 — Domain Authority
# ============================================================

# ============================================================
# Domain Authority Constants
# ============================================================

DOMAIN_AUTHORITY = {
    # Academic / Government — Tier 1 (no ceiling)
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

    # Reputable Health — Tier 2
    "mayoclinic.org":           0.92,
    "medicalnewstoday.com":     0.82,
    "webmd.com":                0.80,
    "healthline.com":           0.80,
    "harvard.edu":              0.95,
    "stanford.edu":             0.95,
    "ox.ac.uk":                 0.93,

    # Quality Tech / AI — Tier 3
    "realpython.com":           0.75,
    "machinelearningmastery.com": 0.72,
    "towardsdatascience.com":   0.70,

    # Open Platforms — Penalized Base + Credential Ceiling
    "medium.com":               0.45,   # ← Penalized base (was 0.55)
    "youtube.com":              0.52,   # ← Penalized base (was 0.65)
    "substack.com":             0.42,   # Similar to Medium
    "wordpress.com":            0.35,   # Lower — more anonymous
    "blogger.com":              0.30,   # Lowest open platform
}

# Maximum score ANY Medium article can achieve
# No matter how credentialed the author
MEDIUM_CEILING  = 0.82
YOUTUBE_CEILING = 0.78

# Medium publications with editorial standards
# These get a boost above base but still below ceiling
MEDIUM_PUBLICATIONS = {
    "towards data science":        0.72,
    "towardsdatascience":          0.72,
    "the startup":                 0.62,
    "thestartup":                  0.62,
    "better humans":               0.62,
    "entrepreneurship handbook":   0.62,
    "mind cafe":                   0.60,
    "tincture":                    0.65,  # Medical publication
    "elemental":                   0.65,  # Health publication
    "personal growth":             0.57,
    "lifestyle":                   0.55,
    "careers":                     0.57,
    "productivity":                0.60,
}


def score_domain_authority(
    domain: str,
    content: str = "",
    source_type: str = "blog",
    author_score: float = 0.0,
) -> float:
    """
    Score domain authority with platform-aware penalties.

    Open Platform Logic:
      Medium base = 0.45 (penalized — no editorial oversight)
      Boosts available from:
        → Author credibility (max +0.20)
        → Known publication (max +0.15)
      Hard ceiling = 0.82 (open platform tax always applies)

      This means:
        → Random Medium author: 0.45
        → Medium author with PhD: 0.45 + 0.20 = 0.65
        → PhD author in Tincture: 0.45 + 0.20 + 0.10 = 0.75
        → Best possible Medium: 0.82 (ceiling)
        → Compare: MayoClinic = 0.92 (no ceiling)
        → Gap = 0.10 minimum "open platform tax"

    Args:
        domain:       Clean domain string
        content:      Article text (for publication detection)
        source_type:  blog / youtube / pubmed
        author_score: Author credibility score (for boost calc)

    Returns:
        Float in [0.0, 1.0]
    """
    domain = domain.lower().replace("www.", "")

    # ── PubMed always 1.0 ─────────────────────────────────
    if "ncbi.nlm.nih.gov" in domain or "pubmed" in domain:
        return 1.00

    # ── Academic domains ───────────────────────────────────
    if domain.endswith(".edu"):
        return 0.88
    if ".ac." in domain:
        return 0.85
    if domain.endswith(".gov"):
        return 0.90

    # ── Direct known domain lookup ─────────────────────────
    base_score = DOMAIN_AUTHORITY.get(domain, 0.38)

    # ── Medium: penalized base + author boost + ceiling ───
    if domain == "medium.com":
        return _score_medium_domain(
            base_score  = base_score,   # 0.45
            content     = content,
            author_score= author_score,
        )

    # ── YouTube: penalized base + channel boost + ceiling ─
    if domain == "youtube.com":
        return _score_youtube_domain(
            base_score   = base_score,  # 0.52
            author_score = author_score,
        )

    # ── Substack / WordPress / Blogger ────────────────────
    if domain in ("substack.com", "wordpress.com", "blogger.com"):
        # Author boost possible but smaller ceiling
        if author_score >= 0.80:
            boosted = base_score + 0.15
        elif author_score >= 0.60:
            boosted = base_score + 0.08
        else:
            boosted = base_score
        return round(min(boosted, 0.70), 3)

    # ── Unknown domain ─────────────────────────────────────
    if base_score == 0.38:
        print(f"[DOMAIN] Unknown domain: {domain} → base 0.38")

    return round(base_score, 3)


def _score_medium_domain(
    base_score: float,
    content: str,
    author_score: float,
) -> float:
    """
    Medium domain scoring with explicit penalty structure.

    Penalty logic:
      Base 0.45 = penalized (open platform, no editorial board)
      Boost 1: Author credentials → up to +0.20
      Boost 2: Known publication → up to +0.15
      Boost 3: High engagement proxy → up to +0.05
      Ceiling: 0.82 = open platform tax always applies

    Why 0.82 ceiling?
      The best Medium article can still only be:
        "A credentialed person writing on an open platform"
      It can never be:
        "An institutionally reviewed, editorially vetted source"
      That 0.18 gap (from 1.0) is the "open platform tax"
    """
    boost = 0.0

    # Boost 1: Author credibility
    if author_score >= 0.85:
        boost += 0.20
        print(f"[DOMAIN] Medium: author boost +0.20 (score={author_score:.2f})")
    elif author_score >= 0.70:
        boost += 0.12
        print(f"[DOMAIN] Medium: author boost +0.12 (score={author_score:.2f})")
    elif author_score >= 0.55:
        boost += 0.06
        print(f"[DOMAIN] Medium: author boost +0.06 (score={author_score:.2f})")
    else:
        print(f"[DOMAIN] Medium: no author boost (score={author_score:.2f})")

    # Boost 2: Known publication
    if content:
        pub_boost = _detect_medium_publication(content)
        if pub_boost > 0:
            boost += pub_boost
            print(f"[DOMAIN] Medium: publication boost +{pub_boost:.2f}")

    raw_score = base_score + boost
    final     = round(min(raw_score, MEDIUM_CEILING), 3)

    print(
        f"[DOMAIN] Medium: base={base_score} + boost={boost:.2f} "
        f"= {raw_score:.3f} → capped at {final} "
        f"(ceiling={MEDIUM_CEILING})"
    )

    return final


def _score_youtube_domain(
    base_score: float,
    author_score: float,
) -> float:
    """
    YouTube domain scoring with penalty structure.

    Base 0.52: Open platform (higher than Medium because
    video format has some natural quality filter — harder to
    produce than a blog post, but still no editorial review).

    Ceiling 0.78: Lower than Medium ceiling because:
    → No citations possible in video format
    → Harder to verify claims
    → Transcripts may be auto-generated
    """
    boost = 0.0

    if author_score >= 0.85:
        boost += 0.18
    elif author_score >= 0.70:
        boost += 0.10
    elif author_score >= 0.55:
        boost += 0.05

    raw   = base_score + boost
    final = round(min(raw, YOUTUBE_CEILING), 3)

    print(
        f"[DOMAIN] YouTube: base={base_score} + boost={boost:.2f} "
        f"= {raw:.3f} → capped at {final} "
        f"(ceiling={YOUTUBE_CEILING})"
    )

    return final


def _detect_medium_publication(content: str) -> float:
    """
    Detect known Medium publication in content.
    Returns boost amount (above base), not the final score.
    """
    content_lower = content.lower()
    best_boost = 0.0

    for pub_name, pub_score in MEDIUM_PUBLICATIONS.items():
        if pub_name in content_lower:
            boost = pub_score - 0.45  # Boost above Medium base
            if boost > best_boost:
                best_boost = boost
                print(f"[DOMAIN] Medium publication: '{pub_name}' → boost +{boost:.2f}")

    return round(best_boost, 3)

def _detect_medium_publication(content: str) -> float:
    """Detect known Medium publication → return authority boost."""
    content_lower = content.lower()
    best = 0.0
    for pub_name, score in MEDIUM_PUBLICATIONS.items():
        if pub_name in content_lower:
            boost = score - 0.55  # Boost above medium base
            if boost > best:
                best = boost
            print(f"[DOMAIN] Publication '{pub_name}' detected → score {score}")
    return round(best, 3)


# ============================================================
# Variable 4 — Recency
# ============================================================

def score_recency(published_date: str) -> float:
    """
    Score based on age from today.
    Unknown date → mild penalty (0.4)
    """
    if not published_date or published_date in ("Unknown", ""):
        return 0.40

    try:
        pub   = datetime.strptime(published_date, "%Y-%m-%d")
        today = datetime.now()
        age   = (today - pub).days
    except Exception:
        return 0.40

    if age < 0:      return 0.50  # Future date = parsing error
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

MEDICAL_ADVICE_SIGNALS = [
    "you should take", "you should use", "recommended dose",
    "take this medication", "treatment for", "cure for",
    "remedy for", "how to treat", "helps treat",
    "proven to cure", "clinically proven",
    "supplement", "dosage", "side effects",
]

PERSONAL_ESSAY_SIGNALS = [
    "in my experience", "i remember", "i learned",
    "my story", "personal journey", "when i was",
    "i decided", "i felt", "i noticed",
    "from my perspective", "i believe", "in my opinion",
]

MEDICAL_TOPIC_SIGNALS = [
    "health", "medical", "disease", "treatment",
    "diagnosis", "drug", "medication", "therapy",
    "clinical", "symptom", "nutrition", "diet",
    "mental health", "depression", "anxiety",
    "cancer", "diabetes", "obesity", "vaccine",
    "supplement", "exercise", "fitness", "wellness",
    "nervous system", "brain", "psychology",
    "neuroscience", "cognitive", "behavior",
]

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


def score_medical_disclaimer(
    content: str,
    topic_tags: list,
    has_disclaimer: bool = False,
) -> tuple:
    """
    Score medical disclaimer presence.

    Key fix: Distinguish between types of medical content.

    Types:
      1. Personal essay / experience piece
         → Author shares their story
         → Disclaimer less critical (0.5 neutral)
         → Even without disclaimer

      2. Medical advice article
         → "Take X for Y condition"
         → "Here is how to treat Z"
         → Disclaimer CRITICAL
         → Without disclaimer → 0.0

      3. Non-medical content
         → Tech, career, lifestyle
         → Disclaimer not applicable → 0.5 neutral

    Returns:
        (score: float, is_medical: bool)
    """
    content_lower = content.lower()
    tags_lower    = " ".join(topic_tags).lower()
    combined      = f"{content_lower} {tags_lower}"

    # Check for disclaimer first
    has_disclaimer_content = has_disclaimer or any(
        phrase in content_lower
        for phrase in DISCLAIMER_PHRASES
    )

    # Is this medical topic at all?
    medical_signal_count = sum(
        1 for signal in MEDICAL_TOPIC_SIGNALS
        if signal in combined
    )
    is_medical = medical_signal_count >= 2

    if not is_medical:
        return 0.50, False   # Not applicable → neutral

    # Is this a personal essay? (less strict)
    personal_essay_count = sum(
        1 for signal in PERSONAL_ESSAY_SIGNALS
        if signal in content_lower
    )
    is_personal_essay = personal_essay_count >= 3

    # Is this direct medical advice? (most strict)
    advice_signal_count = sum(
        1 for signal in MEDICAL_ADVICE_SIGNALS
        if signal in content_lower
    )
    is_medical_advice = advice_signal_count >= 2

    if has_disclaimer_content:
        return 1.00, True

    if is_medical_advice:
        # Direct medical advice without disclaimer → strong penalty
        return 0.00, True

    if is_personal_essay:
        # Personal essay about medical topic → mild penalty
        return 0.40, True

    # Medical content, no advice, no essay signal → moderate penalty
    return 0.20, True


# ============================================================
# Abuse Detection
# ============================================================

def compute_abuse_multiplier(
    author: str,
    content: str,
    domain: str,
    topic_tags: list,
    metadata: dict = None,
) -> tuple:
    """
    Enhanced abuse detection covering ALL assignment requirements:

    1. Fake Authors → Cross-check name patterns
    2. SEO Spam    → Low authority + keyword stuffing
    3. Misleading Medical → No disclaimer + medical claims
    4. Outdated Info → Strong recency penalty (already in recency score)

    Returns:
        (multiplier: float [0.3-1.0], issues: list[str])
    """
    multiplier = 1.0
    issues     = []
    metadata   = metadata or {}

    # ══════════════════════════════════════════════════════
    # CHECK 1: FAKE AUTHORS
    # Cross-check author names with known patterns
    # ══════════════════════════════════════════════════════

    if author and author not in ("Unknown Author", ""):
        author_lower = author.lower()

        # Gibberish: low vowel ratio
        letters = [c for c in author_lower if c.isalpha()]
        if letters:
            vowel_ratio = sum(
                1 for c in letters if c in "aeiou"
            ) / len(letters)
            if vowel_ratio < 0.10:
                multiplier -= 0.25
                issues.append(
                    "FAKE AUTHOR: Name appears to be gibberish "
                    f"(vowel ratio: {vowel_ratio:.2f})"
                )

        # Numbers in name (bot pattern)
        if re.search(r'\d{3,}', author):
            multiplier -= 0.20
            issues.append("FAKE AUTHOR: Suspicious numbers in name")

        # Very short name (single character)
        if len(author.strip()) < 3:
            multiplier -= 0.20
            issues.append("FAKE AUTHOR: Name too short")

        # All caps (spam signal)
        if author.isupper() and len(author) > 4:
            multiplier -= 0.10
            issues.append("FAKE AUTHOR: Name is all caps")

        # Known fake patterns
        fake_patterns = [
            r'^user\d+$', r'^admin\d*$', r'^test\d*$',
            r'^guest\d*$', r'^bot\d*$',
        ]
        for pattern in fake_patterns:
            if re.match(pattern, author_lower):
                multiplier -= 0.30
                issues.append(f"FAKE AUTHOR: Matches bot pattern '{pattern}'")
                break

        # Cross-check: medical credential claimed but no bio
        name_has_cred = any(
            re.search(r'\b' + re.escape(c) + r'\b', author_lower)
            for c in NAME_CREDENTIALS
        )
        author_profile = metadata.get("author_profile", {})
        has_bio = bool(author_profile.get("bio", ""))

        if name_has_cred and not has_bio:
            # Credential in name but no supporting bio → suspicious
            # Don't penalize heavily, just flag
            issues.append(
                "AUTHOR NOTE: Credentials claimed in name but "
                "no supporting bio found. Cannot fully verify."
            )

    # ══════════════════════════════════════════════════════
    # CHECK 2: SEO SPAM BLOGS
    # Penalize domains with low authority + spam signals
    # ══════════════════════════════════════════════════════

    if content:
        words = content.lower().split()
        total_words = len(words)

        if total_words > 0:
            # Keyword stuffing: any word > 5% of content
            word_freq = {}
            for w in words:
                if len(w) > 4:
                    word_freq[w] = word_freq.get(w, 0) + 1

            for word, count in word_freq.items():
                freq = count / total_words
                if freq > 0.05:
                    multiplier -= 0.15
                    issues.append(
                        f"SEO SPAM: Keyword stuffing detected — "
                        f"'{word}' appears {freq:.1%} of content"
                    )
                    break

            # Very thin content on low-authority domain
            da = DOMAIN_AUTHORITY.get(domain, 0.45)
            if total_words < 200 and da < 0.60:
                multiplier -= 0.15
                issues.append(
                    f"SEO SPAM: Thin content ({total_words} words) "
                    f"on low-authority domain ({domain})"
                )

    # Domain spam patterns
    SPAM_DOMAINS = [
        r'\d{4,}',          # Numbers in domain
        r'free.*money',     # Financial spam
        r'click.*here',     # Clickbait
        r'best.*review',    # Fake review sites
    ]
    domain_lower = domain.lower()
    for pattern in SPAM_DOMAINS:
        if re.search(pattern, domain_lower):
            multiplier -= 0.15
            issues.append(f"SEO SPAM: Suspicious domain pattern: {domain}")
            break

    # YouTube-specific spam
    desc_analysis = metadata.get("description_analysis", {})
    spam_count = desc_analysis.get("spam_signals", 0)
    if spam_count >= 3:
        multiplier -= 0.15
        issues.append(
            f"SEO SPAM: {spam_count} promotional phrases in description"
        )

    # ══════════════════════════════════════════════════════
    # CHECK 3: MISLEADING MEDICAL CONTENT
    # Medical claims without disclaimer = dangerous
    # ══════════════════════════════════════════════════════

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

    content_lower = content.lower() if content else ""
    for phrase in MISINFORMATION_PHRASES:
        if phrase in content_lower:
            multiplier -= 0.30
            issues.append(
                f"MISLEADING MEDICAL: Misinformation signal: '{phrase}'"
            )
            break

    # Medical claims + no disclaimer
    has_disclaimer = metadata.get("has_medical_disclaimer", False)
    medical_advice_signals = [
        "you should take", "recommended dose",
        "take this medication", "helps treat",
        "proven to cure", "clinically proven",
    ]
    has_medical_advice = any(
        s in content_lower for s in medical_advice_signals
    )
    if has_medical_advice and not has_disclaimer:
        multiplier -= 0.20
        issues.append(
            "MISLEADING MEDICAL: Direct medical advice "
            "without disclaimer"
        )

    # ══════════════════════════════════════════════════════
    # CHECK 4: OUTDATED INFORMATION
    # Already handled by recency_score variable
    # But add extra flag for very old content making current claims
    # ══════════════════════════════════════════════════════

    # This is handled in score_recency() with strong penalties
    # for 5+ year old content. No additional multiplier needed
    # because it would double-penalize.

    return max(0.30, round(multiplier, 2)), issues

# ============================================================
# Main Trust Score
# ============================================================

def compute_trust_score(article: dict) -> dict:
    """
    Compute trust score for a single scraped article.

    Args:
        article: Dict from scraper

    Returns:
        Dict with trust_score + full breakdown
    """
    source_type = article.get("source_type", "blog")
    metadata    = article.get("metadata", {})
    content     = " ".join(article.get("content_chunks", []))
    topic_tags  = article.get("topic_tags", [])
    domain      = metadata.get("domain", "")

    author      = article.get("author", "Unknown Author")
    pub_date    = article.get("published_date", "Unknown")

    # Blog-specific signals
    author_profile = metadata.get("author_profile", {})
    bio            = author_profile.get("bio", "")
    followers      = author_profile.get("followers", 0)
    article_count  = author_profile.get("article_count", 0)
    clap_count     = metadata.get("clap_count", 0)

    # PubMed-specific signals
    citation_count = metadata.get("citation_count", 0)
    has_disclaimer = metadata.get("has_medical_disclaimer", False)

    print(f"\n{'='*55}")
    print(f"[TRUST] {article.get('title', '')[:50]}")
    print(f"[TRUST] Author: {author}")
    print(f"[TRUST] Source: {source_type} | Domain: {domain}")
    print(f"{'='*55}")

    # ── Variable 1: Author Credibility ────────────────────
    v1_author, author_signals = score_author_credibility(
        author        = author,
        bio           = bio,
        followers     = followers,
        article_count = article_count,
        source_type   = source_type,
        content       = content,
    )

    # ── Variable 2: Citation / Engagement ─────────────────
    v2_citation = score_citation_count(
        citation_count = citation_count,
        clap_count     = clap_count,
        source_type    = source_type,
    )

    # ── Variable 3: Domain Authority ──────────────────────
    # Pass author score so credentialed authors boost domain
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
        author     = author,
        content    = content,
        domain     = domain,
        topic_tags = topic_tags,
    )

    # ── Final Score ───────────────────────────────────────
    final_score = round(
        max(0.0, min(1.0, weighted_sum * abuse_multiplier)),
        3
    )

    # ── Pretty Print ──────────────────────────────────────
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

    return {
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
            },
        },
        "abuse_detection": {
            "multiplier": abuse_multiplier,
            "issues":     abuse_issues,
        },
        "weighted_sum":  round(weighted_sum, 4),
        "final_score":   final_score,
    }


def score_all(articles: list) -> list:
    """Score all articles. Updates trust_score field in each."""
    for article in articles:
        breakdown = compute_trust_score(article)
        article["trust_score"]           = breakdown["final_score"]
        article["trust_score_breakdown"] = breakdown
    return articles


# ============================================================
# Direct Runner
# ============================================================

if __name__ == "__main__":
    import json

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

        with open(fpath) as f:
            articles = json.load(f)

        scored = score_all(articles)
        all_results.extend(scored)

        with open(fpath, "w") as f:
            json.dump(scored, f, indent=2, ensure_ascii=False)

        print(f"\n[SAVED] {fname} updated with trust scores")

    # Summary
    print(f"\n{'='*70}")
    print(f"  TRUST SCORE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Title':<45} {'Type':<8} {'Score':>6}  Grade")
    print(f"{'-'*70}")

    for a in all_results:
        title  = (a.get("title") or "Unknown")[:44]
        stype  = a.get("source_type", "?")[:7]
        score  = a.get("trust_score", 0) or 0

        if score >= 0.80:   grade = "🟢 High Trust"
        elif score >= 0.60: grade = "🟡 Moderate"
        elif score >= 0.40: grade = "🟠 Low"
        else:               grade = "🔴 Unreliable"

        print(f"{title:<45} {stype:<8} {score:>6.3f}  {grade}")
    print()