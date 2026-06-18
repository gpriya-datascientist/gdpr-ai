"""
Layer: DOMAIN
Imports allowed: stdlib + domain.models only
Purpose: Abstract interfaces (ports) — infrastructure implements these
"""
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from domain.models import (
    AuditEntry, ClassifiedQuery, Document,
    LLMResponse, Query, RetrievedContext,
    SanitizedQuery, SensitivityLevel,
)


class IDocumentParser(ABC):
    @abstractmethod
    def parse(self, filepath: str) -> Document: ...


class IChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[str]: ...


class IEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_single(self, text: str) -> list[float]: ...


class IVectorStore(ABC):
    @abstractmethod
    def upsert(self, document: Document, embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        document_id: Optional[UUID] = None,
    ) -> RetrievedContext: ...

    @abstractmethod
    def delete_document(self, document_id: UUID) -> None: ...  # GDPR Art.17


class IPIIDetector(ABC):
    @abstractmethod
    def detect_and_mask(self, query: Query) -> SanitizedQuery: ...

    @abstractmethod
    def scan_text(self, text: str) -> bool:
        """Returns True if PII is found — used as gate check."""
        ...


class IIntentClassifier(ABC):
    @abstractmethod
    def classify(self, query: Query, document: Optional[Document] = None) -> ClassifiedQuery: ...


class ILLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_tokens: int = 500,
    ) -> LLMResponse: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


class IAuditLogger(ABC):
    @abstractmethod
    def log(self, entry: AuditEntry) -> UUID: ...

    @abstractmethod
    def get_entries(self, session_id: str) -> list[AuditEntry]: ...

    @abstractmethod
    def delete_by_session(self, session_id: str) -> int:
        """GDPR Art.17 right to erasure — returns count deleted."""
        ...


class IEncryptor(ABC):
    @abstractmethod
    def encrypt(self, plaintext: str) -> str: ...

    @abstractmethod
    def decrypt(self, ciphertext: str) -> str: ...


class ISemanticCache(ABC):
    @abstractmethod
    def get(self, query_text: str) -> Optional[LLMResponse]: ...

    @abstractmethod
    def set(self, query_text: str, response: LLMResponse) -> None: ...

    @abstractmethod
    def invalidate(self, query_text: str) -> None: ...
