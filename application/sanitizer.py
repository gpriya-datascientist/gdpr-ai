"""
Layer: APPLICATION
Imports allowed: domain only
Purpose: PII sanitization service — wraps IPIIDetector interface.
         This is the GDPR gate. Nothing goes to cloud without passing here.
"""
import logging
from uuid import UUID

from domain.exceptions import PIILeakError
from domain.interfaces import IPIIDetector
from domain.models import Query, SanitizedQuery

logger = logging.getLogger(__name__)


class PIISanitizationService:
    """
    Application-layer service that enforces the sanitization gate.

    Architecture guarantee:
        - sanitize() ALWAYS runs before any cloud call
        - If PII is detected, PIILeakError is raised — cloud call is aborted
        - The sanitized text (not original) is what reaches cloud providers
    """

    def __init__(self, detector: IPIIDetector) -> None:
        self._detector = detector

    def sanitize(self, query: Query) -> SanitizedQuery:
        """
        Run PII detection and masking.
        Raises PIILeakError if PII is found and cloud routing was intended.
        """
        result = self._detector.detect_and_mask(query)

        if result.pii_detected:
            logger.warning(
                "PII detected in query %s: %d entities masked. Cloud call blocked.",
                query.id, len(result.masks_applied),
            )
            # Log each masked type for audit
            for mask in result.masks_applied:
                logger.debug("  Masked [%s]: '%s' → '%s'",
                             mask.pii_type.value, mask.original[:20], mask.masked)

        return result

    def verify_safe_for_cloud(self, sanitized: SanitizedQuery) -> None:
        """
        Hard gate: raises PIILeakError if sanitized query is not safe.
        Called immediately before every cloud LLM call.
        This is the final enforcement point — no exceptions.
        """
        if not sanitized.safe_for_cloud or sanitized.pii_detected:
            raise PIILeakError(
                pii_count=len(sanitized.masks_applied),
                query_id=str(sanitized.original_query.id),
            )
        logger.info("Sanitization gate PASSED for query %s", sanitized.original_query.id)

    def quick_scan(self, text: str) -> bool:
        """Returns True if PII detected — for pre-checks."""
        return self._detector.scan_text(text)
