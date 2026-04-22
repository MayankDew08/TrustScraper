# config.py
# ============================================================
# Central configuration — all keys, weights, constants here
# Never hardcode these in scraper files
# ============================================================

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):

    # ----------------------------
    # API Keys
    # ----------------------------
    groq_api_key: str = ""
    gemini_api_key: str = ""
    youtube_api_key: str = ""
    ncbi_api_key: str = ""          # Optional, increases rate limit

    # ----------------------------
    # Scraper Settings
    # ----------------------------
    request_timeout: int = 15       # Seconds before giving up
    max_retries: int = 3            # Retry failed requests
    retry_delay: float = 2.0        # Seconds between retries
    request_delay: float = 1.5      # Polite delay between requests

    # ----------------------------
    # Chunking Settings
    # ----------------------------
    chunk_size: int = 500           # Max words per chunk
    chunk_overlap: int = 50         # Word overlap between chunks

    # ----------------------------
    # Trust Score Weights
    # Must sum to 1.0
    # ----------------------------
    weight_author_credibility: float = 0.25
    weight_citation_count: float = 0.20
    weight_domain_authority: float = 0.25
    weight_recency: float = 0.15
    weight_medical_disclaimer: float = 0.15

    # ----------------------------
    # Tagging Settings
    # ----------------------------
    max_tags: int = 6               # Max topic tags per article
    keyphrase_ngram_min: int = 1
    keyphrase_ngram_max: int = 2

    # ----------------------------
    # LLM Settings
    # ----------------------------
    groq_model: str = "llama-3.1-70b-versatile"
    gemini_model: str = "gemini-1.5-flash"
    max_tokens: int = 1024
    temperature: float = 0.3        # Low = more factual responses

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance — import this everywhere
settings = Settings()