"""
Layer: APPLICATION
Imports allowed: domain only
Purpose: Classify query sensitivity and determine routing.
         Uses zero-shot scoring + rule-based fallback for reliability.
"""
import logging
import re
from typing import Optional

from domain.interfaces import IIntentClassifier
from domain.models import (
    ClassifiedQuery, Document, Query,
    RouteDecision, SensitivityLevel,
)

logger = logging.getLogger(__name__)

# Keywords that strongly indicate sensitive/local-only processing
SENSITIVE_SIGNALS = [
    "contract", "vertrag", "salary", "gehalt", "medical", "medizin",
    "patient", "invoice", "rechnung", "personal", "hr", "employee",
    "mitarbeiter", "confidential", "vertraulich", "cv", "lebenslauf",
    "audit", "compliance", "gdpr", "dsgvo", "password", "passwort",
    "address", "adresse", "account", "konto", "document", "dokument",
    "certificate", "zertifikat", "signature", "unterschrift",
]

# Keywords that indicate general knowledge (safe for cloud)
GENERAL_SIGNALS = [
    "what is", "was ist", "explain", "erkläre", "how does", "wie funktioniert",
    "definition", "describe", "difference between", "unterschied zwischen",
    "best practice", "example", "beispiel", "tutorial", "overview",
    "legislation", "law", "gesetz", "regulation", "verordnung",
    "history", "geschichte", "concept", "konzept",
]

# Mixed query patterns — always treat as sensitive
MIXED_QUERY_PATTERNS = [
    r"(what|explain|describe).+(in this|my|the uploaded|this|from the)",
    r"(summarize|zusammenfassen).+(document|doc|file|contract|report)",
    r"(translate|übersetzen).+(this|the|my)",
    r"(improve|verbessern|fix|korrigieren).+(paragraph|section|clause)",
    r"(compare|vergleichen).+(this|my|uploaded)",
]


class RuleBasedIntentClassifier(IIntentClassifier):
    """
    Intent classifier combining:
    1. Document-context detection (if doc uploaded → always local)
    2. Keyword signal scoring
    3. Mixed-query pattern detection
    4. Confidence-based fallback to sensitive (safe default)

    Design principle: when in doubt → LOCAL_ONLY.
    A false positive (treat general as sensitive) costs latency.
    A false negative (treat sensitive as general) causes GDPR violation.
    """

    def __init__(
        self,
        sensitivity_threshold: float = 0.7,
        injection_patterns: Optional[list[str]] = None,
    ) -> None:
        self._threshold = sensitivity_threshold
        self._mixed_re = [re.compile(p, re.IGNORECASE) for p in MIXED_QUERY_PATTERNS]
        self._injection_patterns = injection_patterns or []

    def classify(
        self,
        query: Query,
        document: Optional[Document] = None,
    ) -> ClassifiedQuery:
        text_lower = query.raw_text.lower()

        # Rule 1: Document context always → local only
        if document is not None or query.document_id is not None:
            return self._make_decision(
                query, SensitivityLevel.HIGH, RouteDecision.LOCAL_ONLY,
                confidence=1.0, reasoning="Document context present — local only",
            )

        # Rule 2: Mixed query patterns → local only
        for pattern in self._mixed_re:
            if pattern.search(query.raw_text):
                return self._make_decision(
                    query, SensitivityLevel.HIGH, RouteDecision.LOCAL_ONLY,
                    confidence=0.95,
                    reasoning=f"Mixed query pattern matched: {pattern.pattern}",
                )

        # Rule 3: Score sensitive vs general signals
        sensitive_score = self._score_signals(text_lower, SENSITIVE_SIGNALS)
        general_score = self._score_signals(text_lower, GENERAL_SIGNALS)

        total = sensitive_score + general_score
        if total == 0:
            # No signals — default to sensitive (safe)
            return self._make_decision(
                query, SensitivityLevel.MEDIUM, RouteDecision.LOCAL_ONLY,
                confidence=0.5, reasoning="No signals detected — defaulting to local",
            )

        sensitivity_ratio = sensitive_score / total

        if sensitivity_ratio >= self._threshold:
            return self._make_decision(
                query, SensitivityLevel.HIGH, RouteDecision.LOCAL_ONLY,
                confidence=sensitivity_ratio,
                reasoning=f"Sensitive signal score {sensitivity_ratio:.2f}",
            )

        if sensitivity_ratio <= (1 - self._threshold):
            return self._make_decision(
                query, SensitivityLevel.LOW, RouteDecision.CLOUD_SANITIZED,
                confidence=1 - sensitivity_ratio,
                reasoning=f"General signal score {1 - sensitivity_ratio:.2f}",
            )

        # Ambiguous — default to local
        return self._make_decision(
            query, SensitivityLevel.MEDIUM, RouteDecision.LOCAL_ONLY,
            confidence=0.5,
            reasoning=f"Ambiguous sensitivity ratio {sensitivity_ratio:.2f} — local default",
        )

    @staticmethod
    def _score_signals(text: str, signals: list[str]) -> float:
        return sum(1.0 for s in signals if s in text)

    @staticmethod
    def _make_decision(
        query: Query,
        sensitivity: SensitivityLevel,
        route: RouteDecision,
        confidence: float,
        reasoning: str,
    ) -> ClassifiedQuery:
        logger.info(
            "Query %s classified: %s → %s (confidence=%.2f) reason='%s'",
            query.id, sensitivity.value, route.value, confidence, reasoning,
        )
        return ClassifiedQuery(
            query=query,
            sensitivity=sensitivity,
            route=route,
            confidence=confidence,
            reasoning=reasoning,
        )
