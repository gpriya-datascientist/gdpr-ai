"""
Layer: APPLICATION
Imports allowed: domain only
Purpose: RAG pipeline — retrieve relevant chunks and generate local answer.
"""
import logging
from typing import Optional
from uuid import UUID

from domain.interfaces import IEmbedder, ILLMClient, IVectorStore
from domain.models import Document, LLMResponse, Query, RetrievedContext

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are a secure document assistant.
Answer the user's question using ONLY the context provided below.
Do not use any external knowledge. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""


class RAGService:
    """
    Local RAG pipeline:
    1. Embed query
    2. Retrieve top-k relevant chunks from local vector store
    3. Generate answer using local LLM only

    All processing stays on-machine. No external calls.
    """

    def __init__(
        self,
        embedder: IEmbedder,
        vector_store: IVectorStore,
        local_llm: ILLMClient,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._local_llm = local_llm
        self._top_k = top_k

    def ingest_document(self, document: Document, chunks: list[str]) -> None:
        """Embed and store document chunks in local vector store."""
        document.chunks = chunks
        embeddings = self._embedder.embed(chunks)
        self._vector_store.upsert(document, embeddings)
        logger.info("Ingested document %s: %d chunks", document.id, len(chunks))

    def retrieve(
        self,
        query: Query,
        document_id: Optional[UUID] = None,
    ) -> RetrievedContext:
        """Retrieve relevant chunks for a query."""
        query_embedding = self._embedder.embed_single(query.raw_text)
        context = self._vector_store.query(
            embedding=query_embedding,
            top_k=self._top_k,
            document_id=document_id,
        )
        logger.info(
            "Retrieved %d chunks for query %s (top score: %.3f)",
            len(context.chunks),
            query.id,
            max(context.scores, default=0.0),
        )
        return context

    def generate(self, query: Query, context: RetrievedContext) -> LLMResponse:
        """Generate answer from retrieved context using local LLM."""
        context_text = "\n\n".join(
            f"[Chunk {i+1}]\n{chunk}"
            for i, chunk in enumerate(context.chunks)
        )
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_text,
            question=query.raw_text,
        )
        response = self._local_llm.generate(prompt=prompt)
        logger.info(
            "RAG generation complete: %d tokens, %.1fms",
            response.tokens_used, response.latency_ms,
        )
        return response

    def answer(
        self,
        query: Query,
        document_id: Optional[UUID] = None,
    ) -> LLMResponse:
        """Full RAG pipeline: retrieve + generate."""
        context = self.retrieve(query, document_id)
        if not context.chunks:
            return LLMResponse(
                text="I could not find relevant information in the uploaded document.",
                model="rag",
                provider="local",
            )
        return self.generate(query, context)

    def delete_document(self, document_id: UUID) -> None:
        """GDPR Art.17 — remove document from vector store."""
        self._vector_store.delete_document(document_id)
        logger.warning("GDPR erasure: removed document %s from vector store", document_id)
