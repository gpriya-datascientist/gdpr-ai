"""
Layer: INFRASTRUCTURE
Imports allowed: domain + sentence-transformers + chromadb
Purpose: Local embeddings and ChromaDB vector store — fully offline.
"""
import logging
from typing import Optional
from uuid import UUID

from domain.exceptions import VectorStoreError
from domain.interfaces import IEmbedder, IVectorStore
from domain.models import Document, RetrievedContext

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(IEmbedder):
    """MiniLM-L6-v2 — fast, local, good quality. 384-dim embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info("Embedder loaded: %s", model_name)
        except ImportError as e:
            raise RuntimeError("sentence-transformers not installed") from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_single(self, text: str) -> list[float]:
        return self._model.encode([text], convert_to_numpy=True)[0].tolist()


class ChromaVectorStore(IVectorStore):
    """
    Local persistent ChromaDB store.
    Each document gets its own collection for clean GDPR erasure.
    """

    def __init__(self, persist_dir: str) -> None:
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info("ChromaDB initialized at %s", persist_dir)
        except ImportError as e:
            raise RuntimeError("chromadb not installed") from e

    def _collection_name(self, document_id: UUID) -> str:
        return f"doc_{str(document_id).replace('-', '_')}"

    def _global_collection(self):
        return self._client.get_or_create_collection("global_chunks")

    def upsert(self, document: Document, embeddings: list[list[float]]) -> None:
        try:
            col = self._client.get_or_create_collection(
                self._collection_name(document.id)
            )
            col.upsert(
                ids=[f"{document.id}_{i}" for i in range(len(document.chunks))],
                embeddings=embeddings,
                documents=document.chunks,
                metadatas=[{"doc_id": str(document.id), "chunk": i}
                           for i in range(len(document.chunks))],
            )
            logger.info("Upserted %d chunks for doc %s", len(document.chunks), document.id)
        except Exception as e:
            raise VectorStoreError(f"Upsert failed: {e}") from e

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        document_id: Optional[UUID] = None,
    ) -> RetrievedContext:
        try:
            if document_id:
                col = self._client.get_or_create_collection(
                    self._collection_name(document_id)
                )
            else:
                col = self._global_collection()

            results = col.query(
                query_embeddings=[embedding],
                n_results=min(top_k, col.count()),
            )
            chunks = results["documents"][0] if results["documents"] else []
            scores = [1 - d for d in (results["distances"][0] if results["distances"] else [])]
            return RetrievedContext(
                chunks=chunks,
                scores=scores,
                source_document_id=document_id,
            )
        except Exception as e:
            raise VectorStoreError(f"Query failed: {e}") from e

    def delete_document(self, document_id: UUID) -> None:
        """GDPR Art.17 — delete entire document collection."""
        try:
            self._client.delete_collection(self._collection_name(document_id))
            logger.warning("GDPR erasure: deleted collection for doc %s", document_id)
        except Exception as e:
            raise VectorStoreError(f"Delete failed: {e}") from e
