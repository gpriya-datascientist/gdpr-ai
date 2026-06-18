"""
Layer: CONFIG (cross-cutting)
Purpose: Dependency injection container.
         THE ONLY place where concrete implementations are wired together.
         All other modules receive interfaces — never concrete classes.
"""
from functools import lru_cache

from config.settings import CloudProvider, Settings, get_settings


def build_pii_detector(settings: Settings):
    from infrastructure.security.pii_detector import PresidioPIIDetector
    return PresidioPIIDetector(
        threshold=settings.presidio_threshold,
        block_on_uncertainty=settings.pii_block_on_uncertainty,
    )


def build_audit_logger(settings: Settings):
    from infrastructure.security.audit_logger import SQLiteAuditLogger
    return SQLiteAuditLogger(db_path=settings.audit_db_path)


def build_encryptor(settings: Settings):
    from infrastructure.security.encryptor import FernetEncryptor
    return FernetEncryptor(key=settings.encryption_key)


def build_embedder(settings: Settings):
    from infrastructure.rag.embedder_store import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(model_name=settings.embedding_model)


def build_vector_store(settings: Settings):
    from infrastructure.rag.embedder_store import ChromaVectorStore
    return ChromaVectorStore(persist_dir=settings.chroma_persist_dir)


def build_chunker(settings: Settings):
    from infrastructure.parsers.document_parser import SlidingWindowChunker
    return SlidingWindowChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )


def build_document_parser():
    from infrastructure.parsers.document_parser import DocumentParser
    return DocumentParser()


def build_local_llm(settings: Settings):
    from infrastructure.llm.ollama_client import OllamaClient
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )


def build_cloud_llm(settings: Settings):
    """Returns None if cloud is disabled — local-only mode."""
    if settings.cloud_provider == CloudProvider.NONE:
        return None
    if settings.cloud_provider == CloudProvider.GROQ:
        from infrastructure.llm.groq_client import GroqClient
        return GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)
    if settings.cloud_provider == CloudProvider.OPENAI:
        from infrastructure.llm.groq_client import GroqClient  # swap for OpenAI client
        return GroqClient(api_key=settings.openai_api_key, model=settings.openai_model)
    return None


def build_sanitizer(settings: Settings):
    from application.sanitizer import PIISanitizationService
    detector = build_pii_detector(settings)
    return PIISanitizationService(detector=detector)


def build_classifier(settings: Settings):
    from application.classifier import RuleBasedIntentClassifier
    return RuleBasedIntentClassifier(
        sensitivity_threshold=settings.sensitivity_threshold,
        injection_patterns=settings.injection_patterns,
    )


def build_rag_service(settings: Settings):
    from application.rag_service import RAGService
    return RAGService(
        embedder=build_embedder(settings),
        vector_store=build_vector_store(settings),
        local_llm=build_local_llm(settings),
        top_k=settings.retrieval_top_k,
    )


@lru_cache
def build_orchestrator():
    """
    Fully wired orchestrator — call this from API routes.
    Cached so infrastructure is initialized only once per process.
    """
    from application.orchestrator import Orchestrator
    settings = get_settings()
    return Orchestrator(
        classifier=build_classifier(settings),
        sanitizer=build_sanitizer(settings),
        rag_service=build_rag_service(settings),
        local_llm=build_local_llm(settings),
        cloud_llm=build_cloud_llm(settings),
        audit_logger=build_audit_logger(settings),
        cache=None,  # add Redis cache in production
    )
