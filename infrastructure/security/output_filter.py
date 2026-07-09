"""
Layer: INFRASTRUCTURE — Security
Purpose: Complete output sanitization filter.
         Gap 1 fix: Now also runs Presidio NER on output to catch names.
         Gap 2 fix: Rate limiting added.
         Scans every LLM response BEFORE it reaches the user.
"""
import logging
import re
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

OUTPUT_PII_PATTERNS = [
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "EMAIL"),
    (r"\b4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b", "CREDIT_CARD"),
    (r"\b5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b", "CREDIT_CARD"),
    (r"\bDE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b", "IBAN"),
    (r"\b(\+49|0049)\s?(\d{3,5})\s?(\d{6,8})\b", "DE_PHONE"),
    (r"\+\d{1,3}[\s\-]?\d{2,4}[\s\-]?\d{4,10}\b", "INTL_PHONE"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "US_PHONE"),
    (r"\b[A-Z]{1,2}\d{7}\b", "ID_NUMBER"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
]

RECONSTRUCTION_PHRASES = [
    r"the (credit card|card|account) number (is|would be|appears to be)",
    r"the (first|last) (four|six|eight) digits (of the card|of the number)",
    r"(visa|mastercard|amex|american express) card number",
    r"the email address (is|was|would be)\s+\w+@",
    r"the (actual|real) (name|phone|address) (is|was|would be)",
]

# Rate limiting store: session_id -> list of timestamps
_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX = 30       # max requests
RATE_LIMIT_WINDOW = 60    # per 60 seconds


class OutputSanitizationFilter:
    """
    Complete output sanitization filter — 3 layers:
    1. Regex PII patterns (email, card, IBAN, phone, SSN)
    2. Presidio NER on output (catches names the model reconstructed)
    3. Reconstruction phrase detection (blocks "the card number is...")
    """

    def __init__(self) -> None:
        self._pii_patterns = [(re.compile(p, re.IGNORECASE), label)
                               for p, label in OUTPUT_PII_PATTERNS]
        self._reconstruction_patterns = [re.compile(p, re.IGNORECASE)
                                          for p in RECONSTRUCTION_PHRASES]
        self._analyzer = self._init_presidio()

    def _init_presidio(self):
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            })
            return AnalyzerEngine(nlp_engine=provider.create_engine())
        except Exception as e:
            logger.warning("Output filter Presidio init failed: %s", e)
            return None

    def check_rate_limit(self, session_id: str) -> bool:
        """Returns True if request is allowed, False if rate limited."""
        now = time.time()
        timestamps = _rate_store[session_id]
        # Remove old timestamps outside window
        _rate_store[session_id] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        if len(_rate_store[session_id]) >= RATE_LIMIT_MAX:
            logger.warning("Rate limit exceeded for session %s", session_id)
            return False
        _rate_store[session_id].append(now)
        return True

    def filter(self, response_text: str, original_query: Optional[str] = None,
               masked_names: Optional[list[str]] = None) -> tuple[str, int]:
        """
        Filter LLM response.
        Returns (filtered_text, count_of_items_redacted).
        masked_names: list of original names that were masked in input
        """
        if not response_text:
            return response_text, 0

        # Layer 1: Reconstruction phrase detection — block entire response
        for pattern in self._reconstruction_patterns:
            if pattern.search(response_text):
                logger.warning("OUTPUT FILTER: Reconstruction attempt — blocking response")
                return ("That information has been redacted for privacy protection.", 1)

        # Layer 2: Regex PII scan
        redacted_count = 0
        filtered = response_text
        for pattern, label in self._pii_patterns:
            matches = pattern.findall(filtered)
            if matches:
                filtered = pattern.sub(f"[{label}_REDACTED]", filtered)
                redacted_count += len(matches)

        # Layer 3: Presidio NER on output — catches reconstructed names
        if self._analyzer:
            try:
                results = self._analyzer.analyze(
                    text=filtered, language="en", score_threshold=0.7,
                    entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
                )
                for r in sorted(results, key=lambda x: x.start, reverse=True):
                    entity_text = filtered[r.start:r.end]
                    filtered = filtered[:r.start] + f"[{r.entity_type}_REDACTED]" + filtered[r.end:]
                    redacted_count += 1
                    logger.warning("OUTPUT FILTER (NER): Redacted %s from response", r.entity_type)
            except Exception as e:
                logger.warning("Output NER scan failed: %s", e)

        if redacted_count > 0:
            logger.warning("OUTPUT FILTER: Total %d items redacted from response", redacted_count)

        return filtered, redacted_count
