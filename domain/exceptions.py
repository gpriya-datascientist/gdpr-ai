"""
Layer: DOMAIN
Imports allowed: stdlib only
Purpose: Domain exceptions — raised by application, caught by interfaces
"""


class EuroSecError(Exception):
    """Base exception for all EuroSec AI errors."""


class PIILeakError(EuroSecError):
    """Raised when PII is detected in a query destined for cloud."""
    def __init__(self, pii_count: int, query_id: str):
        self.pii_count = pii_count
        self.query_id = query_id
        super().__init__(
            f"GDPR VIOLATION BLOCKED: {pii_count} PII entities "
            f"detected in query {query_id}. Cloud call aborted."
        )


class RouteViolationError(EuroSecError):
    """Raised when routing logic detects an attempted bypass."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Routing violation detected: {reason}")


class PromptInjectionError(EuroSecError):
    """Raised when adversarial injection is detected in input."""
    def __init__(self, pattern: str):
        self.pattern = pattern
        super().__init__(f"Prompt injection pattern detected: {pattern}")


class DocumentParseError(EuroSecError):
    """Raised when a document cannot be parsed."""


class EncryptionError(EuroSecError):
    """Raised when encryption or decryption fails."""


class LLMClientError(EuroSecError):
    """Raised when an LLM provider call fails."""
    def __init__(self, provider: str, reason: str):
        self.provider = provider
        super().__init__(f"LLM client error [{provider}]: {reason}")


class VectorStoreError(EuroSecError):
    """Raised when vector store operations fail."""


class AuditLogError(EuroSecError):
    """Raised when audit logging fails — this is critical."""
