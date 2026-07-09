"""
Layer: APPLICATION
Purpose: Classify query sensitivity and determine routing.
         Fixed: GDPR general questions route to cloud, not local.
         Fixed: No-signal queries default to cloud (fast).
         Fixed: Lower threshold so more queries use fast Groq.
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
# NOTE: removed "gdpr", "address", "personal", "document" — these are general topics
SENSITIVE_SIGNALS = [
    "my name is", "my email", "my phone", "my address", "my password",
    "my credit card", "my iban", "my account", "my salary", "my id",
    "vertrag", "gehalt", "medizin", "patient", "rechnung",
    "mitarbeiter", "vertraulich", "lebenslauf", "passwort",
    "konto", "zertifikat", "unterschrift",
    "invoice number", "account number", "social security",
    "date of birth", "passport", "driving licence",
]

# Keywords that indicate general knowledge (safe for cloud) — expanded
GENERAL_SIGNALS = [
    "what is", "what are", "was ist", "explain", "how does", "how do",
    "wie funktioniert", "definition", "describe", "difference between",
    "best practice", "example", "tutorial", "overview", "summarise",
    "legislation", "law", "regulation", "history", "concept",
    "gdpr", "dsgvo", "article", "principle", "right", "compliance",
    "data minimisation", "data protection", "privacy", "consent",
    "controller", "processor", "data subject", "lawful basis",
    "tell me", "can you", "please explain", "what day", "today",
    "industrial", "manufacturing", "assembly", "sensor", "stroke rate",
]

MIXED_QUERY_PATTERNS = [
    r"(summarize|zusammenfassen).+(my|this uploaded|this|from the)",
    r"(translate|übersetzen).+(this|the|my)",
    r"(improve|fix).+(my|this paragraph|my clause)",
    r"(compare).+(my|this uploaded)",
]


class RuleBasedIntentClassifier(IIntentClassifier):
    """
    Fixed classifier:
    - General knowledge questions → CLOUD_SANITIZED (fast Groq)
    - Queries with personal data signals → LOCAL_ONLY (private Mistral)
    - Unknown/no signals → CLOUD_SANITIZED (fast, PII detector will catch any PII)
    """

    def __init__(
        self,
        sensitivity_threshold: float = 0.6,
        injection_patterns: Optional[list[str]] = None,
    ) -> None:
        self._threshold = sensitivity_threshold
        self._mixed_re = [re.compile(p, re.IGNORECASE) for p in MIXED_QUERY_PATTERNS]

    def classify(self, query: Query, document: Optional[Document] = None) -> ClassifiedQuery:
        text_lower = query.raw_text.lower()

        # Rule 1: Document context → local only
        if document is not None or query.document_id is not None:
            return self._decide(query, SensitivityLevel.HIGH, RouteDecision.LOCAL_ONLY,
                                1.0, "Document context — local only")

        # Rule 2: Mixed query patterns → local only
        for pattern in self._mixed_re:
            if pattern.search(query.raw_text):
                return self._decide(query, SensitivityLevel.HIGH, RouteDecision.LOCAL_ONLY,
                                    0.95, f"Mixed query pattern: {pattern.pattern}")

        # Rule 3: Score signals
        sensitive_score = sum(1.0 for s in SENSITIVE_SIGNALS if s in text_lower)
        general_score   = sum(1.0 for s in GENERAL_SIGNALS   if s in text_lower)

        # Has sensitive signals → local
        if sensitive_score > 0 and sensitive_score >= general_score:
            return self._decide(query, SensitivityLevel.HIGH, RouteDecision.LOCAL_ONLY,
                                0.9, f"Sensitive signals: {sensitive_score}")

        # Has general signals OR no signals → cloud (fast)
        # PII detector will still catch any personal data before it leaves
        return self._decide(query, SensitivityLevel.LOW, RouteDecision.CLOUD_SANITIZED,
                            0.85, f"General query — routing to cloud (general={general_score})")

    @staticmethod
    def _decide(query, sensitivity, route, confidence, reasoning) -> ClassifiedQuery:
        logger.info("Query %s → %s (%.2f) — %s", query.id, route.value, confidence, reasoning)
        return ClassifiedQuery(
            query=query, sensitivity=sensitivity, route=route,
            confidence=confidence, reasoning=reasoning,
        )
