# Central app configuration.
# Keep API keys, model settings, and scoring defaults in one place.

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):

    # API keys
    groq_api_key: str = ""
    youtube_api_key: str = ""
    supadata_api_key: str = ""
    ncbi_api_key: str = ""          

    # Scraper behavior
    request_timeout: int = 15       # Timeout per request (seconds)
    max_retries: int = 3            # Retries on transient failures
    retry_delay: float = 2.0        # Delay between retries
    request_delay: float = 1.5      # Friendly pause between calls


    # Chunking behavior
    chunk_size: int = 500           # Max words per chunk
    chunk_overlap: int = 50         # Word overlap between chunks


    # Trust score weights
    # Keep this at 1.0 in total
    weight_author_credibility: float = 0.25
    weight_citation_count: float = 0.20
    weight_domain_authority: float = 0.25
    weight_recency: float = 0.15
    weight_medical_disclaimer: float = 0.15


    # Tagging settings
    max_tags: int = 6               
    keyphrase_ngram_min: int = 1
    keyphrase_ngram_max: int = 2


    # LLM settings
    groq_model: str = "llama-3.1-8b-instant"
    max_tokens: int = 1024
    temperature: float = 0.3        # Lower values keep responses steadier

    # Springer Nature API keys
    springer_meta_api_key: str = ""
    springer_oa_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Shared settings instance used across modules
settings = Settings()