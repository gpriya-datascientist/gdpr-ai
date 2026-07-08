"""
Layer: APPLICATION
Purpose: Main orchestration pipeline — coordinates classify, sanitize, route, generate, audit.
         Now includes output sanitization filter and context isolator.
"""
import logging
import time
from typing import Optional
from uuid import UUID, uuid4

from domain.exceptions import PIILeakError, PromptInjectionError, RouteViolationError
from domain.interfaces import IAuditLogger, IIntentClassifier, ILLMClient, ISemanticCache
from domain.models import (
    AuditEntry, Document, LLMResponse,
    OrchestratorResponse, Query, RouteDecision, SensitivityLevel,
)
from application.sanitizer import PIISanitizationService
from application.rag_service import RAGService
from infrastructure.security.output_filter import OutputSanitizationFilter
from infrastructure.security.context_isolator import ContextIsolator

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        classifier: IIntentClassifier,
        sanitizer: PIISanitizationService,
        rag_service: RAGService,
        local_llm: ILLMClient,
        cloud_llm: Optional[ILLMClient],
        audit_logger: IAuditLogger,
        cache: Optional[ISemanticCache] = None,
    ) -> None:
        self._classifier = classifier
        self._sanitizer = sanitizer
        self._rag = rag_service
        self._local_llm = local_llm
        self._cloud_llm = cloud_llm
        self._audit = audit_logger
        self._cache = cache
        self._output_filter = OutputSanitizationFilter()
        self._context = ContextIsolator()

    def process(self, query: Query, document: Optional[Document] = None) -> OrchestratorResponse:
        start = time.perf_counter()

        # Step 1: Classify intent
        classified = self._classifier.classify(query, document)
        logger.info("Query %s -> route=%s sensitivity=%s confidence=%.2f",
                    query.id, classified.route.value, classified.sensitivity.value, classified.confidence)

        # Step 2: Sanitize regardless of route
        try:
            sanitized = self._sanitizer.sanitize(query)
        except PromptInjectionError as e:
            return self._blocked_response(query, str(e), start)

        # Step 3: Execute route
        llm_response: LLMResponse
        if classified.route == RouteDecision.BLOCKED:
            return self._blocked_response(query, classified.reasoning, start)
        elif classified.route == RouteDecision.LOCAL_ONLY:
            llm_response = self._execute_local(query, document, sanitized.sanitized_text)
        elif classified.route == RouteDecision.CLOUD_SANITIZED:
            llm_response = self._execute_cloud(sanitized, start, query)
        else:
            raise RouteViolationError(f"Unknown route: {classified.route}")

        # Step 4: OUTPUT FILTER — scan LLM response for leaked PII before returning
        filtered_text, output_pii_count = self._output_filter.filter(
            llm_response.text, original_query=query.raw_text
        )
        llm_response.text = filtered_text
        total_pii = len(sanitized.masks_applied) + output_pii_count

        # Step 5: Update context isolator with sanitized turn
        if query.session_id:
            self._context.add_turn(query.session_id, sanitized.sanitized_text, filtered_text)

        # Step 6: Audit log
        latency_ms = (time.perf_counter() - start) * 1000
        audit_id = self._audit.log(AuditEntry(
            query_id=query.id,
            session_id=query.session_id,
            route_decision=classified.route,
            sensitivity_level=classified.sensitivity,
            pii_detected=sanitized.pii_detected,
            pii_count=total_pii,
            provider_called=llm_response.provider,
            gdpr_compliant=True,
            latency_ms=latency_ms,
        ))

        return OrchestratorResponse(
            query_id=query.id,
            answer=filtered_text,
            route_taken=classified.route,
            provider_used=llm_response.provider,
            pii_masked_count=total_pii,
            latency_ms=latency_ms,
            audit_id=audit_id,
        )

    def _execute_local(self, query: Query, document: Optional[Document],
                       sanitized_text: Optional[str] = None) -> LLMResponse:
        safe_prompt = sanitized_text or query.raw_text
        if document or query.document_id:
            return self._rag.answer(query, query.document_id)
        return self._local_llm.generate(prompt=safe_prompt)

    def _execute_cloud(self, sanitized, start, query: Query) -> LLMResponse:
        try:
            self._sanitizer.verify_safe_for_cloud(sanitized)
        except PIILeakError:
            logger.error("PIILeakError: falling back to local for query %s", query.id)
            return self._local_llm.generate(prompt=sanitized.sanitized_text)

        if self._cloud_llm is None:
            return self._local_llm.generate(prompt=sanitized.sanitized_text)

        if self._cache:
            cached = self._cache.get(sanitized.sanitized_text)
            if cached:
                cached.cached = True
                return cached

        response = self._cloud_llm.generate(prompt=sanitized.sanitized_text)
        if self._cache:
            self._cache.set(sanitized.sanitized_text, response)
        return response

    def _blocked_response(self, query: Query, reason: str, start: float) -> OrchestratorResponse:
        latency_ms = (time.perf_counter() - start) * 1000
        self._audit.log(AuditEntry(
            query_id=query.id,
            session_id=query.session_id,
            route_decision=RouteDecision.BLOCKED,
            sensitivity_level=SensitivityLevel.HIGH,
            pii_detected=True,
            pii_count=0,
            provider_called="none",
            gdpr_compliant=True,
            latency_ms=latency_ms,
        ))
        return OrchestratorResponse(
            query_id=query.id,
            answer="This query was blocked for security reasons.",
            route_taken=RouteDecision.BLOCKED,
            provider_used="none",
            pii_masked_count=0,
            latency_ms=latency_ms,
            audit_id=uuid4(),
        )
