"""LangGraph orchestration pipeline.

This module is the CLI entrypoint that wires scraping and scoring
into one graph flow.
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END, START

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper.blog_scraper    import scrape_all_blogs
from scraper.youtube_scraper import scrape_all_youtube
from scraper.pubmed_scraper  import scrape_pubmed
from scoring.trust_score     import score_all


# Shared pipeline state

class PipelineState(TypedDict):
    blog_urls:       list
    youtube_urls:    list
    pubmed_query:    str
    blog_results:    Annotated[list, operator.add]
    youtube_results: Annotated[list, operator.add]
    pubmed_results:  Annotated[list, operator.add]
    scored_results:  list
    errors:          Annotated[list, operator.add]
    started_at:      str
    completed_at:    str
    total_time_secs: float


# Default inputs used by the CLI prompts

DEFAULT_BLOG_URLS = [
    "https://medium.com/@Dr.Shlain/the-long-game-024366b9d61f",
    "https://medium.com/@malynnda.stewart/the-art-of-receiving-feedback-how-to-stay-connected-when-youre-being-criticized-cf75351d842a",
    "https://www.health.harvard.edu/blog/can-men-hold-off-on-treating-recurring-prostate-cancer-202512193108",
]

DEFAULT_YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=nqiuSshC9GA",
    "https://www.youtube.com/watch?v=-dh-QNlX12k",
]

DEFAULT_PUBMED_QUERY = "psychology cognitive behavioral therapy"


# Graph nodes

def input_node(state: PipelineState) -> dict:
    """Validate and log user inputs."""
    print(f"\n{'='*60}")
    print(f"  [NODE] Input Validation")
    print(f"{'='*60}")

    errors = []

    blog_urls = state.get("blog_urls", [])
    valid_blogs = []
    for url in blog_urls:
        if url.startswith("http"):
            valid_blogs.append(url)
            print(f"  ✅ Blog    : {url[:65]}")
        else:
            errors.append(f"Invalid blog URL: {url}")
            print(f"  ❌ Invalid : {url}")

    yt_urls = state.get("youtube_urls", [])
    valid_yt = []
    for url in yt_urls:
        if "youtu" in url or len(url) == 11:
            valid_yt.append(url)
            print(f"  ✅ YouTube : {url[:65]}")
        else:
            errors.append(f"Invalid YouTube URL: {url}")
            print(f"  ❌ Invalid : {url}")

    pubmed_query = state.get("pubmed_query", "").strip()
    if not pubmed_query:
        pubmed_query = DEFAULT_PUBMED_QUERY
        print(f"  ⚠ PubMed  : using default '{pubmed_query}'")
    else:
        print(f"  ✅ PubMed  : '{pubmed_query}'")

    print(f"\n  Blogs   : {len(valid_blogs)}")
    print(f"  YouTube : {len(valid_yt)}")
    print(f"  PubMed  : '{pubmed_query}'")

    return {
        "blog_urls":       valid_blogs,
        "youtube_urls":    valid_yt,
        "pubmed_query":    pubmed_query,
        "errors":          errors,
        "blog_results":    [],
        "youtube_results": [],
        "pubmed_results":  [],
        "scored_results":  [],
        "started_at":      datetime.now(timezone.utc).isoformat(),
    }


def blog_node(state: PipelineState) -> dict:
    """
    Scrape all blog URLs.
    Runs in parallel with youtube_node and pubmed_node.
    """
    print(f"\n{'='*60}")
    print(f"  [NODE] Blog Scraper")
    print(f"{'='*60}")

    blog_urls = state.get("blog_urls", [])

    if not blog_urls:
        print(f"  ⚠ No blog URLs — skipping")
        return {"blog_results": [], "errors": []}

    print(f"  Scraping {len(blog_urls)} blog(s)...")

    try:
        results = scrape_all_blogs(blog_urls)

        success  = sum(
            1 for r in results
            if r.get("metadata", {}).get("chunk_count", 0) > 0
        )
        failed   = len(results) - success

        print(f"\n  [BLOG NODE] ✅ {success}/{len(results)} successful")
        if failed > 0:
            print(f"  [BLOG NODE] ⚠ {failed} failed")

        return {
            "blog_results": results,
            "errors":       [],
        }

    except Exception as e:
        # Print full traceback so we can see exactly what failed
        tb = traceback.format_exc()
        error = f"Blog scraper exception: {e}"
        print(f"\n  [BLOG NODE] ❌ FAILED")
        print(f"  Error: {e}")
        print(f"  Traceback:\n{tb}")
        return {
            "blog_results": [],
            "errors":       [error],
        }


def youtube_node(state: PipelineState) -> dict:
    """
    Scrape YouTube videos.
    Runs in parallel with blog_node and pubmed_node.
    """
    print(f"\n{'='*60}")
    print(f"  [NODE] YouTube Scraper")
    print(f"{'='*60}")

    yt_urls = state.get("youtube_urls", [])

    if not yt_urls:
        print(f"  ⚠ No YouTube URLs — skipping")
        return {"youtube_results": [], "errors": []}

    print(f"  Scraping {len(yt_urls)} video(s)...")

    try:
        results = scrape_all_youtube(yt_urls)
        success = sum(
            1 for r in results
            if r.get("metadata", {}).get("chunk_count", 0) > 0
        )
        print(f"\n  [YT NODE] ✅ {success}/{len(results)} successful")
        return {
            "youtube_results": results,
            "errors":          [],
        }

    except Exception as e:
        tb    = traceback.format_exc()
        error = f"YouTube scraper exception: {e}"
        print(f"\n  [YT NODE] ❌ FAILED: {e}")
        print(f"  Traceback:\n{tb}")
        return {
            "youtube_results": [],
            "errors":          [error],
        }


def pubmed_node(state: PipelineState) -> dict:
    """
    Scrape PubMed article.
    Runs in parallel with blog_node and youtube_node.
    """
    print(f"\n{'='*60}")
    print(f"  [NODE] PubMed Scraper")
    print(f"{'='*60}")

    query = state.get("pubmed_query", "").strip()

    if not query:
        print(f"  ⚠ No query — skipping")
        return {"pubmed_results": [], "errors": []}

    print(f"  Query: '{query}'")

    try:
        result  = scrape_pubmed(query)
        results = [result]
        success = 1 if result.get("metadata", {}).get("chunk_count", 0) > 0 else 0
        print(f"\n  [PUBMED NODE] ✅ {success}/1 successful")
        return {
            "pubmed_results": results,
            "errors":         [],
        }

    except Exception as e:
        tb    = traceback.format_exc()
        error = f"PubMed scraper exception: {e}"
        print(f"\n  [PUBMED NODE] ❌ FAILED: {e}")
        print(f"  Traceback:\n{tb}")
        return {
            "pubmed_results": [],
            "errors":         [error],
        }


def aggregate_node(state: PipelineState) -> dict:
    """Collect results from all parallel scrapers."""
    print(f"\n{'='*60}")
    print(f"  [NODE] Aggregate")
    print(f"{'='*60}")

    blogs   = state.get("blog_results",    [])
    youtube = state.get("youtube_results", [])
    pubmed  = state.get("pubmed_results",  [])
    errors  = state.get("errors",          [])
    total   = len(blogs) + len(youtube) + len(pubmed)

    print(f"  Blogs   : {len(blogs)}")
    print(f"  YouTube : {len(youtube)}")
    print(f"  PubMed  : {len(pubmed)}")
    print(f"  Total   : {total}")

    if errors:
        print(f"\n  ⚠ Errors so far ({len(errors)}):")
        for err in errors:
            print(f"    → {err}")

    if total == 0:
        print(f"  ❌ No results to score — check errors above")
    else:
        print(f"  ✅ Proceeding to trust scoring")

    return {}


def scoring_node(state: PipelineState) -> dict:
    """Compute trust scores for all scraped sources."""
    print(f"\n{'='*60}")
    print(f"  [NODE] Trust Scoring")
    print(f"{'='*60}")

    blogs   = state.get("blog_results",    [])
    youtube = state.get("youtube_results", [])
    pubmed  = state.get("pubmed_results",  [])

    all_articles = blogs + youtube + pubmed

    if not all_articles:
        print(f"  ❌ No articles to score")
        return {"scored_results": []}

    print(f"  Scoring {len(all_articles)} source(s)...")

    try:
        scored = score_all(all_articles)
        print(f"  ✅ Scoring complete")
        return {"scored_results": scored}
    except Exception as e:
        tb    = traceback.format_exc()
        error = f"Scoring exception: {e}"
        print(f"  ❌ Scoring failed: {e}")
        print(f"  Traceback:\n{tb}")
        return {
            "scored_results": all_articles,
            "errors":         [error],
        }


def output_node(state: PipelineState) -> dict:
    """Save results + print summary."""
    print(f"\n{'='*60}")
    print(f"  [NODE] Output")
    print(f"{'='*60}")

    scored = state.get("scored_results", [])
    errors = state.get("errors",         [])

    # Split by source type
    blogs   = [r for r in scored if r.get("source_type") == "blog"]
    youtube = [r for r in scored if r.get("source_type") == "youtube"]
    pubmed  = [r for r in scored if r.get("source_type") == "pubmed"]

    # Save files
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)

    def save(data: list, fname: str) -> str:
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    paths = {
        "blogs":    save(blogs,   "blogs.json"),
        "youtube":  save(youtube, "youtube.json"),
        "pubmed":   save(pubmed,  "pubmed.json"),
        "combined": save(scored,  "scraped_data.json"),
    }

    print(f"\n  Files saved:")
    for name, path in paths.items():
        size = os.path.getsize(path) / 1024
        exists = "✅" if size > 0.1 else "⚠ (empty)"
        print(f"    {name:<10} → {path[-50:]} ({size:.1f} KB) {exists}")

    # Summary table
    print(f"\n{'='*75}")
    print(f"  TRUST SCORE RESULTS")
    print(f"{'='*75}")
    print(
        f"{'#':<3} {'Type':<8} {'Title':<38} "
        f"{'Score':>6}  {'Grade':<14} Chunks"
    )
    print(f"{'-'*75}")

    for i, r in enumerate(scored, 1):
        stype  = r.get("source_type", "?")[:7]
        title  = (r.get("title") or "Unknown")[:37]
        score  = r.get("trust_score") or 0
        chunks = r.get("metadata", {}).get("chunk_count", 0)

        if score >= 0.80:   grade = "🟢 High Trust"
        elif score >= 0.60: grade = "🟡 Moderate"
        elif score >= 0.40: grade = "🟠 Low"
        else:               grade = "🔴 Unreliable"

        print(
            f"{i:<3} {stype:<8} {title:<38} "
            f"{score:>6.3f}  {grade:<14} {chunks}"
        )

    # Stats
    scores = [r.get("trust_score") or 0 for r in scored]
    if scores:
        print(f"\n{'─'*50}")
        print(f"  Total sources    : {len(scored)}")
        print(f"  Average score    : {sum(scores)/len(scores):.3f}")
        print(f"  Highest          : {max(scores):.3f}")
        print(f"  Lowest           : {min(scores):.3f}")
        print(f"  High trust ≥0.80 : {sum(1 for s in scores if s >= 0.80)}")
        print(f"  Moderate 0.60-0.79: {sum(1 for s in scores if 0.60 <= s < 0.80)}")
        print(f"  Low <0.60        : {sum(1 for s in scores if s < 0.60)}")

    # Anomaly flags
    flagged = [
        r for r in scored
        if r.get("ai_explanation", {}).get("anomaly_flag", False)
    ]
    if flagged:
        print(f"\n{'─'*50}")
        print(f"  ⚠ FLAGGED ({len(flagged)})")
        for r in flagged:
            expl = r.get("ai_explanation", {})
            print(f"\n  → {r.get('title','')[:55]}")
            print(f"    Score  : {r.get('trust_score',0):.3f}")
            print(f"    Reason : {expl.get('anomaly_reason','')[:70]}")

    # Pipeline errors
    if errors:
        print(f"\n{'─'*50}")
        print(f"  ⚠ PIPELINE ERRORS ({len(errors)})")
        for err in errors:
            print(f"  → {err}")

    completed_at = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"  Completed : {completed_at}")
    print(f"  Output    : {output_dir}/")
    print(f"{'='*60}\n")

    return {"completed_at": completed_at}


# Build Graph

def build_pipeline():
    """
    Build compiled LangGraph pipeline.

    Graph:
      START → input
                → blogs   (parallel)
                → youtube (parallel)
                → pubmed  (parallel)
              → aggregate
              → scoring
              → output
              → END
    """
    graph = StateGraph(PipelineState)

    graph.add_node("input",     input_node)
    graph.add_node("blogs",     blog_node)
    graph.add_node("youtube",   youtube_node)
    graph.add_node("pubmed",    pubmed_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("scoring",   scoring_node)
    graph.add_node("output",    output_node)

    # Edges
    graph.add_edge(START,       "input")

    # Fan-out: input → parallel scrapers
    graph.add_edge("input",     "blogs")
    graph.add_edge("input",     "youtube")
    graph.add_edge("input",     "pubmed")

    # Fan-in: all scrapers → aggregate
    graph.add_edge("blogs",     "aggregate")
    graph.add_edge("youtube",   "aggregate")
    graph.add_edge("pubmed",    "aggregate")

    # Linear: aggregate → scoring → output → END
    graph.add_edge("aggregate", "scoring")
    graph.add_edge("scoring",   "output")
    graph.add_edge("output",    END)

    return graph.compile()


# CLI Interface

def get_user_inputs() -> dict:
    """
    Interactive CLI — press Enter to use confirmed defaults.
    """
    print("\n" + "=" * 60)
    print("  Multi-Source Scraper + Trust Scoring")
    print("  Powered by LangGraph")
    print("=" * 60)

    # Blog URLs
    print("\n📝 BLOG POSTS (3 URLs)")
    print("  Press Enter to use defaults\n")

    blog_urls = []
    for i, default in enumerate(DEFAULT_BLOG_URLS, 1):
        raw = input(f"  Blog {i} [{default[:50]}...]: ").strip()
        url = raw if raw.startswith("http") else default
        blog_urls.append(url)
        print(f"  → Using: {url[:65]}")

    # YouTube URLs
    print("\n▶ YOUTUBE VIDEOS (2 URLs)")
    print("  Press Enter to use defaults\n")

    yt_urls = []
    for i, default in enumerate(DEFAULT_YOUTUBE_URLS, 1):
        raw = input(f"  YouTube {i} [{default}]: ").strip()
        url = raw if "youtu" in raw else default
        yt_urls.append(url)
        print(f"  → Using: {url}")

    # PubMed
    print(f"\n🔬 PUBMED ARTICLE")
    raw   = input(
        f"  Query [default: '{DEFAULT_PUBMED_QUERY}']: "
    ).strip()
    query = raw if raw else DEFAULT_PUBMED_QUERY
    print(f"  → Using: '{query}'")

    # Confirm
    print(f"\n{'─'*60}")
    print(f"  Ready to scrape:")
    print(f"    Blogs   : {len(blog_urls)}")
    print(f"    YouTube : {len(yt_urls)}")
    print(f"    PubMed  : '{query}'")
    print(f"{'─'*60}")

    confirm = input("\n  Start pipeline? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("  Cancelled.")
        sys.exit(0)

    return {
        "blog_urls":       blog_urls,
        "youtube_urls":    yt_urls,
        "pubmed_query":    query,
        "blog_results":    [],
        "youtube_results": [],
        "pubmed_results":  [],
        "scored_results":  [],
        "errors":          [],
        "started_at":      "",
        "completed_at":    "",
        "total_time_secs": 0.0,
    }


# Entry Point

if __name__ == "__main__":

    pipeline    = build_pipeline()
    init_state  = get_user_inputs()

    print(f"\n▶ Starting pipeline...\n")
    start_time = time.time()

    final_state = pipeline.invoke(
        init_state,
        config={"recursion_limit": 10},
    )

    elapsed = time.time() - start_time
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)")