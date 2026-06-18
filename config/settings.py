"""
Layer: CONFIG (cross-cutting)
Purpose: All environment variables and settings in one place.
         Every other module imports from here — never from os.environ directly.
"""
from functools import lru_cache
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict


class CloudProvider(str, Enum):
    GROQ = "groq"
    OPENAI = "openai"
    NONE = "none"       # local-only mode, no cloud calls at all


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────
    app_name: str = "EuroSec AI"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── Security ─────────────────────────────────────────
    encryption_key: str = ""          # Fernet key — generate with Fernet.generate_key()
    audit_db_path: str = "data/audit.db"
    local_storage_path: str = "data/documents"

    # ── Local LLM ────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b-instruct-q4_K_M"
    ollama_timeout: int = 120

    # ── Cloud provider (only for non-sensitive queries) ──
    cloud_provider: CloudProvider = CloudProvider.GROQ
    groq_api_key: str = ""
    groq_model: str = "llama3-70b-8192"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    max_cloud_tokens: int = 500       # hard cap on cloud response size

    # ── RAG ──────────────────────────────────────────────
    chroma_persist_dir: str = "data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # ── PII detection ────────────────────────────────────
    presidio_threshold: float = 0.6   # confidence threshold for masking
    pii_block_on_uncertainty: bool = True   # safe default: block if unsure

    # ── Intent classification ─────────────────────────────
    sensitivity_threshold: float = 0.7
    injection_patterns: list[str] = [
        "ignore previous", "system prompt", "developer mode",
        "admin override", "disable filter", "cloud override",
        "send to", "forward to", "exfiltrate",
    ]

    # ── Cache ─────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_similarity_threshold: float = 0.92
    cache_ttl_seconds: int = 3600

    # ── MLflow ───────────────────────────────────────────
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "eurosec-ai"

    # ── Token budget ─────────────────────────────────────
    max_history_turns: int = 4        # sliding window for conversation
    max_context_tokens: int = 3000


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()
