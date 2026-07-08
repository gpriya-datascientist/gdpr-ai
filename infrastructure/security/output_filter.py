"""
Layer: INFRASTRUCTURE — Security
Purpose: Output sanitization filter — scans every LLM response BEFORE it reaches the user.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

OUTPUT_PII_PATTERNS = [
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "EMAIL"),
    (r"\b4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b", "CREDIT_CARD"),
    (r"\b5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b", "CREDIT_CARD"),
    (r"\bDE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b", "IBAN"),
    (r"\b(\+49|0049|0)\s?(\d{3,5})\s?(\d{6,8})\b", "DE_PHONE"),
    (r"\+\d{1,3}[\s\-]?\d{2,4}[\s\-]?\d{4,10}\b", "INTL_PHONE"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "US_PHONE"),
    (r"\b[A-Z]{1,2}\d{7}\b", "ID_NUMBER"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
]

RECONSTRUCTION_PHRASES = [
    r"the (name|person) (is|was|would be)",
    r"(card|account) number (is|would be|appears to be)",
    r"the (email|address|phone) (is|was|would be)",
    r"the (first|last) (four|six|eight) digits",
    r"(visa|mastercard|amex|american express) card",
    r"expires? (on|in).*\d{2}[/\-]\d{2,4}",
]


class OutputSanitizationFilter:
    """
    Post-processing filter — scans every LLM response before it reaches the user.
    Three actions: REDACT (replace PII), BLOCK (reconstruction detected), PASS (clean).
    """

    def __init__(self) -> None:
        self._pii_patterns = [(re.compile(p, re.IGNORECASE), label)
                               for p, label in OUTPUT_PII_PATTERNS]
        self._reconstruction_patterns = [re.compile(p, re.IGNORECASE)
                                          for p in RECONSTRUCTION_PHRASES]

    def filter(self, response_text: str, original_query: Optional[str] = None) -> tuple[str, int]:
        """Filter LLM response. Returns (filtered_text, count_of_items_redacted)."""
        if not response_text:
            return response_text, 0

        # Step 1: Check for reconstruction attempts — block entire response
        for pattern in self._reconstruction_patterns:
            if pattern.search(response_text):
                logger.warning("OUTPUT FILTER: Reconstruction attempt detected — blocking")
                return (
                    "That information has been redacted for privacy protection. "
                    "I cannot provide details about protected personal data.",
                    1
                )

        # Step 2: Scan and redact raw PII in response
        redacted_count = 0
        filtered = response_text
        for pattern, label in self._pii_patterns:
            matches = pattern.findall(filtered)
            if matches:
                logger.warning("OUTPUT FILTER: Found %s in response — redacting", label)
                filtered = pattern.sub(f"[{label}_REDACTED]", filtered)
                redacted_count += len(matches)

        if redacted_count > 0:
            logger.warning("OUTPUT FILTER: Redacted %d items from LLM response", redacted_count)

        return filtered, redacted_count
