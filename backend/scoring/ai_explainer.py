# backend/scoring/ai_explainer.py
# ============================================================
# AI-powered trust score explanation
#
# Uses Groq (Llama 3.1) to explain trust scores with
# complete mathematical transparency.
#
# Prompt loaded from prompt.txt — easy to update
# without touching code.
# ============================================================

import os
import sys
import json
import re
from typing import Optional

from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings


# ============================================================
# Prompt Loading
# ============================================================

_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prompt.txt"
)

_PROMPT_TEMPLATE: Optional[str] = None


def _load_prompt_template() -> str:
    """
    Load prompt template from prompt.txt.
    Cached after first load.

    Why file-based prompt?
    → Easy to update without touching Python code
    → Can be version-controlled separately
    → Non-engineers can edit prompt behavior
    → Cleaner codebase
    """
    global _PROMPT_TEMPLATE

    if _PROMPT_TEMPLATE is not None:
        return _PROMPT_TEMPLATE

    try:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            _PROMPT_TEMPLATE = f.read()
        print(f"[PROMPT] Loaded from {_PROMPT_PATH}")
        return _PROMPT_TEMPLATE
    except FileNotFoundError:
        print(f"[PROMPT] ⚠ prompt.txt not found at {_PROMPT_PATH}")
        print(f"[PROMPT] Using minimal fallback prompt")
        _PROMPT_TEMPLATE = _minimal_fallback_prompt()
        return _PROMPT_TEMPLATE


def _minimal_fallback_prompt() -> str:
    """Minimal prompt used if prompt.txt is missing."""
    return """You are a trust scoring analyst. Explain the trust score.

{source_data}

{calculations}

Anomalies: {anomaly_count}
{anomaly_details}

Respond ONLY with valid JSON:
{{
  "summary": "string",
  "mathematical_breakdown": {{
    "author_credibility": "string",
    "citation_count": "string",
    "domain_authority": "string",
    "recency": "string",
    "medical_disclaimer": "string",
    "final_calculation": "string"
  }},
  "key_drivers": ["string", "string", "string"],
  "improvement_suggestions": ["string"],
  "anomaly_flag": false,
  "anomaly_reason": null,
  "verification_questions": []
}}"""


# ============================================================
# Known Authority Thresholds (for anomaly detection)
# ============================================================

HIGH_AUTHORITY_DOMAINS = {
    "health.harvard.edu":      0.82,
    "mayoclinic.org":          0.80,
    "pubmed.ncbi.nlm.nih.gov": 0.88,
    "nature.com":              0.85,
    "who.int":                 0.85,
    "cdc.gov":                 0.85,
    "nih.gov":                 0.85,
    "nejm.org":                0.85,
    "thelancet.com":           0.85,
    "hopkinsmedicine.org":     0.80,
    "clevelandclinic.org":     0.78,
}

LOW_AUTHORITY_DOMAINS = {
    "medium.com":    0.75,
    "youtube.com":   0.78,
    "substack.com":  0.65,
    "wordpress.com": 0.60,
}


# ============================================================
# Anomaly Detection
# ============================================================

def _detect_anomaly(
    score: float,
    breakdown: dict,
    domain: str,
    source_type: str,
    title: str,
    author: str,
) -> dict:
    """
    Rule-based anomaly detection.
    Fast and deterministic — runs before AI call.

    Detects:
      1. High-authority domain scoring below expected minimum
      2. Low-authority domain scoring above expected maximum
      3. Anonymous author on medical content
      4. Medical content with no disclaimer
      5. Large score component mismatch
      6. Abuse signals flagged
    """
    domain_clean = domain.lower().replace("www.", "")
    anomalies    = []

    # ── 1. High authority, low score ──────────────────────
    expected_min = HIGH_AUTHORITY_DOMAINS.get(domain_clean)
    if expected_min and score < expected_min:
        gap = round(expected_min - score, 3)
        anomalies.append({
            "type":     "HIGH_AUTHORITY_LOW_SCORE",
            "severity": "HIGH" if gap > 0.15 else "MEDIUM",
            "reason": (
                f"{domain_clean} is a high-authority institutional source "
                f"(expected minimum score: {expected_min:.2f}) "
                f"but received {score:.3f}. "
                f"Gap of {gap:.3f} suggests a data extraction issue."
            ),
            "verify": [
                f"Is '{author}' a verified {domain_clean} author?",
                "Was the full article content extracted correctly?",
                "Is the publication date correctly parsed?",
                "Are author credentials visible on the author profile page?",
            ],
        })

    # ── 2. Low authority, suspiciously high score ─────────
    expected_max = LOW_AUTHORITY_DOMAINS.get(domain_clean)
    if expected_max and score > expected_max:
        gap = round(score - expected_max, 3)
        anomalies.append({
            "type":     "LOW_AUTHORITY_HIGH_SCORE",
            "severity": "MEDIUM",
            "reason": (
                f"{domain_clean} is an open platform "
                f"(expected maximum: {expected_max:.2f}) "
                f"but scored {score:.3f}. "
                f"Exceeds ceiling by {gap:.3f} — verify author credentials."
            ),
            "verify": [
                f"Are the credentials claimed by '{author}' independently verifiable?",
                "Does the author bio match their claimed expertise?",
                "Are any cited sources real and accessible?",
            ],
        })

    # ── 3. Anonymous medical content ──────────────────────
    is_medical = (
        breakdown.get("medical_disclaimer", {})
        .get("signals", {})
        .get("is_medical", False)
    )
    if author in ("Unknown Author", "", "Unknown") and is_medical:
        anomalies.append({
            "type":     "ANONYMOUS_MEDICAL_CONTENT",
            "severity": "HIGH",
            "reason": (
                "Medical content published without a verifiable author. "
                "Cannot confirm expertise or accountability for medical claims."
            ),
            "verify": [
                "Who is the publisher of this content?",
                "Is there an editorial board or review process listed?",
                "Are the medical claims supported by cited references?",
            ],
        })

    # ── 4. Medical content, no disclaimer ─────────────────
    disclaimer_score = (
        breakdown.get("medical_disclaimer", {}).get("score", 0.5)
    )
    if is_medical and disclaimer_score == 0.0:
        anomalies.append({
            "type":     "MEDICAL_NO_DISCLAIMER",
            "severity": "HIGH",
            "reason": (
                "Medical content detected without a professional disclaimer. "
                "Readers may act on unverified medical advice without being "
                "advised to consult a healthcare professional."
            ),
            "verify": [
                "Does this article contain direct medical advice?",
                "Is there a site-wide disclaimer not found in the article?",
                "Should a warning be displayed to readers?",
            ],
        })

    # ── 5. Score component mismatch ───────────────────────
    components = {
        "author_credibility": breakdown.get(
            "author_credibility", {}
        ).get("score", 0),
        "domain_authority": breakdown.get(
            "domain_authority", {}
        ).get("score", 0),
        "recency": breakdown.get("recency", {}).get("score", 0),
    }
    scores_list = list(components.values())
    if len(scores_list) >= 2:
        score_range = max(scores_list) - min(scores_list)
        if score_range > 0.50:
            low_comp  = min(components, key=components.get)
            high_comp = max(components, key=components.get)
            anomalies.append({
                "type":     "SCORE_COMPONENT_MISMATCH",
                "severity": "LOW",
                "reason": (
                    f"Large spread between {high_comp} "
                    f"({components[high_comp]:.3f}) and {low_comp} "
                    f"({components[low_comp]:.3f}). "
                    f"Range of {score_range:.3f} may indicate incomplete extraction."
                ),
                "verify": [
                    f"Is the {low_comp} data correctly extracted?",
                    "Check the raw JSON output for missing fields.",
                ],
            })

    # ── 6. Abuse signals ───────────────────────────────────
    abuse_issues = (
        breakdown.get("abuse_detection", {}).get("issues", [])
    )
    if abuse_issues:
        anomalies.append({
            "type":     "ABUSE_SIGNALS_DETECTED",
            "severity": "HIGH",
            "reason":   f"Abuse signals: {'; '.join(abuse_issues)}",
            "verify": [
                "Is the author name legitimate?",
                "Is this content original (not scraped/spun)?",
                "Are any medical claims fabricated or exaggerated?",
            ],
        })

    return {
        "is_anomaly":      len(anomalies) > 0,
        "anomaly_count":   len(anomalies),
        "anomalies":       anomalies,
        "highest_severity": (
            "HIGH"   if any(a["severity"] == "HIGH"   for a in anomalies) else
            "MEDIUM" if any(a["severity"] == "MEDIUM" for a in anomalies) else
            "LOW"    if anomalies else
            "NONE"
        ),
    }


# ============================================================
# Data Formatters — Fill prompt.txt placeholders
# ============================================================

def _format_source_data(
    title: str,
    author: str,
    domain: str,
    source_type: str,
    score: float,
    breakdown: dict,
) -> str:
    """Format source metadata section for prompt."""
    author_signals = breakdown.get(
        "author_credibility", {}
    ).get("signals", {})
    tier = author_signals.get("tier", "open_platform")

    grade = (
        "HIGH TRUST (🟢)"   if score >= 0.80 else
        "MODERATE (🟡)"     if score >= 0.60 else
        "LOW TRUST (🟠)"    if score >= 0.40 else
        "UNRELIABLE (🔴)"
    )

    return (
        f"Title       : {title[:100]}\n"
        f"Author      : {author}\n"
        f"Domain      : {domain}\n"
        f"Source Type : {source_type}\n"
        f"Domain Tier : {tier}\n"
        f"Final Score : {score:.4f} / 1.0000 — {grade}"
    )


def _format_calculations(breakdown: dict) -> str:
    """
    Format detailed calculation section for prompt.
    Shows every number used in the formula.
    """
    author_data    = breakdown.get("author_credibility", {})
    citation_data  = breakdown.get("citation_count", {})
    domain_data    = breakdown.get("domain_authority", {})
    recency_data   = breakdown.get("recency", {})
    disclaimer_data= breakdown.get("medical_disclaimer", {})
    abuse_data     = breakdown.get("abuse_detection", {})

    a_score = author_data.get("score", 0)
    a_w     = author_data.get("weight", 0.30)
    a_c     = author_data.get("contribution", 0)

    c_score = citation_data.get("score", 0)
    c_w     = citation_data.get("weight", 0.20)
    c_c     = citation_data.get("contribution", 0)

    d_score = domain_data.get("score", 0)
    d_w     = domain_data.get("weight", 0.20)
    d_c     = domain_data.get("contribution", 0)

    r_score = recency_data.get("score", 0)
    r_w     = recency_data.get("weight", 0.15)
    r_c     = recency_data.get("contribution", 0)

    dis_score = disclaimer_data.get("score", 0)
    dis_w     = disclaimer_data.get("weight", 0.15)
    dis_c     = disclaimer_data.get("contribution", 0)

    multiplier   = abuse_data.get("multiplier", 1.0)
    abuse_issues = abuse_data.get("issues", [])
    weighted_sum = breakdown.get("weighted_sum", 0)
    final_score  = breakdown.get("final_score", 0)

    # Author signals
    author_signals   = author_data.get("signals", {})
    credentials      = author_signals.get("credentials_found", [])
    followers        = author_signals.get("followers", 0)
    citations_found  = author_signals.get("citations_in_content", 0)
    institution_floor= author_signals.get("institution_floor", 0)
    tier             = author_signals.get("tier", "open_platform")

    # Citation signals
    citation_signals = citation_data.get("signals", {})
    clap_count       = citation_signals.get("clap_count", 0)
    citation_count   = citation_signals.get("citation_count", 0)
    domain_tier_cit  = citation_signals.get("domain_tier", "open")

    # Recency signals
    recency_signals  = recency_data.get("signals", {})
    pub_date         = recency_signals.get("published_date", "Unknown")

    # Disclaimer signals
    dis_signals      = disclaimer_data.get("signals", {})
    is_medical       = dis_signals.get("is_medical", False)
    has_disclaimer   = dis_signals.get("has_disclaimer", False)
    domain_tier_dis  = dis_signals.get("domain_tier", "open")

    lines = [
        "COMPONENT 1: AUTHOR CREDIBILITY",
        f"  Score      : {a_score:.4f}",
        f"  Weight     : {a_w}",
        f"  Contribution: {a_score:.4f} × {a_w} = {a_c:.4f}",
        f"  Tier       : {tier}",
    ]

    if institution_floor:
        lines.append(f"  Floor score: {institution_floor} (institutional guarantee)")
    else:
        lines.append(f"  Base score : 0.25 (open platform start)")

    lines += [
        f"  Credentials: {credentials if credentials else 'None found'}",
        f"  Followers  : {followers:,}",
        f"  Citations  : {citations_found} found in content",
        "",
        "COMPONENT 2: CITATION / ENGAGEMENT",
        f"  Score      : {c_score:.4f}",
        f"  Weight     : {c_w}",
        f"  Contribution: {c_score:.4f} × {c_w} = {c_c:.4f}",
        f"  Domain tier: {domain_tier_cit}",
    ]

    if domain_tier_cit == "institutional":
        lines.append(f"  Method     : Neutral 0.60 (institutional editorial process)")
    elif clap_count > 0:
        import math
        calc = math.log10(clap_count + 1) / 4.0
        lines.append(f"  Clap count : {clap_count:,}")
        lines.append(f"  Formula    : log10({clap_count} + 1) / 4.0 = {calc:.4f}")
    elif citation_count > 0:
        import math
        calc = math.log10(citation_count + 1) / 2.0
        lines.append(f"  Citations  : {citation_count}")
        lines.append(f"  Formula    : log10({citation_count} + 1) / 2.0 = {calc:.4f}")
    else:
        lines.append(f"  No engagement data → base 0.10")

    lines += [
        "",
        "COMPONENT 3: DOMAIN AUTHORITY",
        f"  Score      : {d_score:.4f}",
        f"  Weight     : {d_w}",
        f"  Contribution: {d_score:.4f} × {d_w} = {d_c:.4f}",
        f"  Domain tier: {'institutional (hardcoded score)' if tier == 'institutional' else 'open platform (base 0.45 + author boost)'}",
        "",
        "COMPONENT 4: RECENCY",
        f"  Score      : {r_score:.4f}",
        f"  Weight     : {r_w}",
        f"  Contribution: {r_score:.4f} × {r_w} = {r_c:.4f}",
        f"  Published  : {pub_date}",
        "",
        "COMPONENT 5: MEDICAL DISCLAIMER",
        f"  Score      : {dis_score:.4f}",
        f"  Weight     : {dis_w}",
        f"  Contribution: {dis_score:.4f} × {dis_w} = {dis_c:.4f}",
        f"  Is medical : {is_medical}",
        f"  Disclaimer : {has_disclaimer}",
        f"  Domain tier: {domain_tier_dis}",
        "",
        "ABUSE MULTIPLIER",
        f"  Multiplier : {multiplier}",
        f"  Issues     : {abuse_issues if abuse_issues else 'None'}",
        "",
        "FINAL CALCULATION",
        f"  ({a_c:.4f} + {c_c:.4f} + {d_c:.4f} + {r_c:.4f} + {dis_c:.4f}) × {multiplier}",
        f"  = {weighted_sum:.4f} × {multiplier}",
        f"  = {final_score:.4f}",
    ]

    return "\n".join(lines)


def _format_anomaly_details(anomaly_report: dict) -> str:
    """Format anomaly details section for prompt."""
    anomalies = anomaly_report.get("anomalies", [])

    if not anomalies:
        return "No anomalies detected."

    lines = []
    for a in anomalies:
        lines.append(
            f"[{a['severity']}] {a['type']}\n"
            f"  Reason : {a['reason']}\n"
            f"  Verify : {'; '.join(a.get('verify', []))}"
        )

    return "\n\n".join(lines)


# ============================================================
# Main Public Function
# ============================================================

def generate_trust_explanation(
    title: str,
    author: str,
    domain: str,
    source_type: str,
    score: float,
    breakdown: dict,
) -> dict:
    """
    Generate AI explanation for a trust score.

    Pipeline:
      1. Load prompt template from prompt.txt
      2. Rule-based anomaly detection
      3. Format data sections
      4. Fill prompt template placeholders
      5. Send to Groq (Llama 3.1 70B)
      6. Parse JSON response
      7. Merge with anomaly data
      8. Return structured explanation

    Falls back to rule-based explanation if Groq unavailable.
    """
    print(f"\n[AI EXPLAIN] '{title[:50]}'")

    # ── Step 1: Load prompt ────────────────────────────────
    template = _load_prompt_template()

    # ── Step 2: Anomaly detection ──────────────────────────
    anomaly_report = _detect_anomaly(
        score       = score,
        breakdown   = breakdown,
        domain      = domain,
        source_type = source_type,
        title       = title,
        author      = author,
    )

    if anomaly_report["is_anomaly"]:
        print(
            f"[AI EXPLAIN] ⚠ {anomaly_report['anomaly_count']} anomaly/anomalies "
            f"({anomaly_report['highest_severity']} severity)"
        )

    # ── Step 3: Format data sections ──────────────────────
    source_data      = _format_source_data(
        title, author, domain, source_type, score, breakdown
    )
    calculations     = _format_calculations(breakdown)
    anomaly_details  = _format_anomaly_details(anomaly_report)
    anomaly_count    = str(anomaly_report["anomaly_count"])

    # ── Step 4: Fill prompt template ───────────────────────
    prompt = (
        template
        .replace("{source_data}",     source_data)
        .replace("{calculations}",    calculations)
        .replace("{anomaly_count}",   anomaly_count)
        .replace("{anomaly_details}", anomaly_details)
    )

    # ── Step 5: Call Groq ──────────────────────────────────
    if not settings.groq_api_key:
        print(f"[AI EXPLAIN] No Groq key → rule-based fallback")
        explanation = _rule_based_explanation(
            score          = score,
            breakdown      = breakdown,
            anomaly_report = anomaly_report,
            domain         = domain,
            author         = author,
        )
    else:
        try:
            client = Groq(api_key=settings.groq_api_key)

            response = client.chat.completions.create(
                model       = settings.groq_model,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.2,
                max_tokens  = 700,
            )

            raw = response.choices[0].message.content.strip()
            print(f"[AI EXPLAIN] ✅ Response: {len(raw)} chars")

            explanation = _parse_ai_response(raw)

        except Exception as e:
            print(f"[AI EXPLAIN] ⚠ Groq failed: {e} → fallback")
            explanation = _rule_based_explanation(
                score          = score,
                breakdown      = breakdown,
                anomaly_report = anomaly_report,
                domain         = domain,
                author         = author,
            )

    # ── Step 6: Merge anomaly data ─────────────────────────
    if anomaly_report["is_anomaly"] and not explanation.get("anomaly_flag"):
        explanation["anomaly_flag"]           = True
        explanation["anomaly_reason"]         = anomaly_report["anomalies"][0]["reason"]
        explanation["verification_questions"] = anomaly_report["anomalies"][0].get("verify", [])

    explanation["anomaly_report"] = anomaly_report

    # ── Step 7: Print summary ─────────────────────────────
    print(f"[AI EXPLAIN] Summary: {explanation.get('summary', '')[:100]}...")
    if explanation.get("anomaly_flag"):
        print(f"[AI EXPLAIN] ⚠ FLAGGED: {explanation.get('anomaly_reason', '')[:80]}")
        for q in explanation.get("verification_questions", []):
            print(f"  → {q}")

    return explanation


# ============================================================
# Response Parser
# ============================================================

def _parse_ai_response(raw: str) -> dict:
    """Parse JSON from AI response. Handles markdown code blocks."""
    clean = re.sub(r'```(?:json)?\s*', '', raw).strip()
    clean = re.sub(r'```\s*$', '', clean).strip()

    default = {
        "summary":                "",
        "mathematical_breakdown": {},
        "key_drivers":            [],
        "improvement_suggestions": [],
        "anomaly_flag":           False,
        "anomaly_reason":         None,
        "verification_questions": [],
    }

    try:
        data = json.loads(clean)
        return {
            "summary":                 data.get("summary", ""),
            "mathematical_breakdown":  data.get("mathematical_breakdown", {}),
            "key_drivers":             data.get("key_drivers", []),
            "improvement_suggestions": data.get("improvement_suggestions", []),
            "anomaly_flag":            bool(data.get("anomaly_flag", False)),
            "anomaly_reason":          data.get("anomaly_reason"),
            "verification_questions":  data.get("verification_questions", []),
        }
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return {
                    "summary":                 data.get("summary", ""),
                    "mathematical_breakdown":  data.get("mathematical_breakdown", {}),
                    "key_drivers":             data.get("key_drivers", []),
                    "improvement_suggestions": data.get("improvement_suggestions", []),
                    "anomaly_flag":            bool(data.get("anomaly_flag", False)),
                    "anomaly_reason":          data.get("anomaly_reason"),
                    "verification_questions":  data.get("verification_questions", []),
                }
            except Exception:
                pass

    default["summary"] = raw[:300]
    return default


# ============================================================
# Rule-Based Fallback
# ============================================================

def _rule_based_explanation(
    score: float,
    breakdown: dict,
    anomaly_report: dict,
    domain: str,
    author: str,
) -> dict:
    """Fallback when Groq unavailable — deterministic explanation."""
    import math

    a_data  = breakdown.get("author_credibility", {})
    c_data  = breakdown.get("citation_count", {})
    d_data  = breakdown.get("domain_authority", {})
    r_data  = breakdown.get("recency", {})
    dis_data= breakdown.get("medical_disclaimer", {})
    abuse   = breakdown.get("abuse_detection", {})

    a_score  = a_data.get("score", 0)
    a_w      = a_data.get("weight", 0.30)
    a_c      = a_data.get("contribution", 0)
    c_score  = c_data.get("score", 0)
    c_w      = c_data.get("weight", 0.20)
    c_c      = c_data.get("contribution", 0)
    d_score  = d_data.get("score", 0)
    d_w      = d_data.get("weight", 0.20)
    d_c      = d_data.get("contribution", 0)
    r_score  = r_data.get("score", 0)
    r_w      = r_data.get("weight", 0.15)
    r_c      = r_data.get("contribution", 0)
    dis_score= dis_data.get("score", 0)
    dis_w    = dis_data.get("weight", 0.15)
    dis_c    = dis_data.get("contribution", 0)
    mult     = abuse.get("multiplier", 1.0)
    wsum     = breakdown.get("weighted_sum", 0)

    a_signals  = a_data.get("signals", {})
    creds      = a_signals.get("credentials_found", [])
    followers  = a_signals.get("followers", 0)
    tier       = a_signals.get("tier", "open_platform")
    inst_floor = a_signals.get("institution_floor", 0)

    r_signals  = r_data.get("signals", {})
    pub_date   = r_signals.get("published_date", "Unknown")

    dis_signals  = dis_data.get("signals", {})
    is_medical   = dis_signals.get("is_medical", False)
    has_dis      = dis_signals.get("has_disclaimer", False)

    grade = (
        "HIGH TRUST"   if score >= 0.80 else
        "MODERATE"     if score >= 0.60 else
        "LOW TRUST"    if score >= 0.40 else
        "UNRELIABLE"
    )

    # Summary
    top_contrib = max(
        [("author_credibility", a_c),
         ("domain_authority",   d_c),
         ("recency",            r_c)],
        key=lambda x: x[1]
    )
    summary = (
        f"This {domain} source scores {score:.3f}/1.000 ({grade}). "
        f"The primary driver is {top_contrib[0]} "
        f"(contributing {top_contrib[1]:.4f} to total). "
        f"{'Institutional backing provides a guaranteed credibility floor.' if tier == 'institutional' else 'Score reflects open platform with author-based credibility.'}"
    )

    # Math breakdown
    math_bd = {
        "author_credibility": (
            f"Scored {a_score:.4f} × {a_w} = {a_c:.4f}. "
            f"{'Floor ' + str(inst_floor) + ' from ' + domain + ' institutional tier. ' if inst_floor else 'Base 0.25 open platform. '}"
            f"Credentials: {creds if creds else 'none'}. "
            f"Followers: {followers:,}."
        ),
        "citation_count": (
            f"Scored {c_score:.4f} × {c_w} = {c_c:.4f}. "
            f"{'Institutional neutral 0.60 — editorial quality implied.' if tier == 'institutional' else 'Based on engagement data.'}"
        ),
        "domain_authority": (
            f"Scored {d_score:.4f} × {d_w} = {d_c:.4f}. "
            f"Domain {domain} — {'institutional tier, direct lookup.' if tier == 'institutional' else 'open platform, base 0.45 + author boost.'}"
        ),
        "recency": (
            f"Scored {r_score:.4f} × {r_w} = {r_c:.4f}. "
            f"Published {pub_date}."
        ),
        "medical_disclaimer": (
            f"Scored {dis_score:.4f} × {dis_w} = {dis_c:.4f}. "
            f"Medical: {is_medical}. Disclaimer present: {has_dis}."
        ),
        "final_calculation": (
            f"({a_c:.4f} + {c_c:.4f} + {d_c:.4f} + {r_c:.4f} + {dis_c:.4f}) "
            f"× {mult} = {wsum:.4f} × {mult} = {score:.4f}"
        ),
    }

    # Key drivers
    contribs = [
        ("author_credibility", a_c),
        ("citation_count",     c_c),
        ("domain_authority",   d_c),
        ("recency",            r_c),
        ("medical_disclaimer", dis_c),
    ]
    contribs.sort(key=lambda x: x[1], reverse=True)
    key_drivers = [
        f"{name}: {score_val:.4f} score × weight = {contrib:.4f} contribution"
        for name, contrib in contribs[:3]
        for score_val in [breakdown.get(name, {}).get("score", 0)]
    ]

    # Improvement suggestions
    suggestions = []
    if is_medical and not has_dis and dis_score < 0.85:
        gain = round(dis_w * (0.85 - dis_score), 4)
        suggestions.append(
            f"Add medical disclaimer: disclaimer 0.00 → 0.85, gain +{gain:.4f}"
        )
    if a_score < 0.70 and tier != "institutional":
        gain = round(a_w * (0.70 - a_score), 4)
        suggestions.append(
            f"Verifiable author credentials: author score → 0.70, gain +{gain:.4f}"
        )
    if not suggestions:
        suggestions.append("Score is appropriate for this source type.")

    # Anomaly data
    anomaly_flag   = anomaly_report.get("is_anomaly", False)
    anomaly_reason = None
    verify_qs      = []
    if anomaly_flag and anomaly_report.get("anomalies"):
        anomaly_reason = anomaly_report["anomalies"][0]["reason"]
        verify_qs      = anomaly_report["anomalies"][0].get("verify", [])

    return {
        "summary":                 summary,
        "mathematical_breakdown":  math_bd,
        "key_drivers":             key_drivers,
        "improvement_suggestions": suggestions,
        "anomaly_flag":            anomaly_flag,
        "anomaly_reason":          anomaly_reason,
        "verification_questions":  verify_qs,
    }