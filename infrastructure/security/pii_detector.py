"""
Layer: INFRASTRUCTURE
Imports allowed: domain + presidio + spacy
Purpose: PII detection and masking using Microsoft Presidio.
         Extended with German-specific entities (IBAN, Steuernummer, Personalausweis).
"""
import logging
import re
from typing import Optional

from domain.exceptions import PIILeakError
from domain.interfaces import IPIIDetector
from domain.models import PIIMask, PIIType, Query, SanitizedQuery

logger = logging.getLogger(__name__)

# German-specific PII patterns not covered by Presidio defaults
GERMAN_PATTERNS = {
    PIIType.FINANCIAL: [
        (r"\bDE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b", "IBAN"),
        # Credit card patterns (Visa, Mastercard, Amex, etc.)
        (r"\b4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b", "CREDIT_CARD"),
        (r"\b5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b", "CREDIT_CARD"),
        (r"\b3[47][0-9]{2}[\s\-]?[0-9]{6}[\s\-]?[0-9]{5}\b", "CREDIT_CARD"),
        # Card expiry dates
        (r"\bexpir[yed]{0,2}\s*:?\s*(0[1-9]|1[0-2])[\s\/\-]([0-9]{2}|[0-9]{4})\b", "CARD_EXPIRY"),
        (r"\b(0[1-9]|1[0-2])[\s\/\-]([0-9]{2}|[0-9]{4})\b(?=\s*(cvv|cvc|expir|$))", "CARD_EXPIRY"),
    ],
    PIIType.ID_NUMBER: [
        (r"\b[A-Z]{1,2}\d{7}\b", "Personalausweis"),
        (r"\b\d{2,3}/\d{3}/\d{5}\b", "Steuernummer"),
        (r"\b\d{11}\b", "Steuer-ID"),
        # CVV/CVC codes
        (r"\b(cvv|cvc|cvv2|cvc2)[\s:]*\d{3,4}\b", "CARD_CVV"),
    ],
    PIIType.PHONE: [
        (r"\b(\+49|0049|0)\s?(\d{3,5})\s?(\d{6,8})\b", "DE_PHONE"),
        # International phone formats
        (r"\+\d{1,3}[\s\-]?\d{2,4}[\s\-]?\d{4,10}\b", "INTL_PHONE"),
    ],
    PIIType.ADDRESS: [
        (r"\b\d{5}\s+[A-Za-z\u00c0-\u00ff][a-z\u00c0-\u00ff]+\b", "DE_POSTCODE_CITY"),
        # Street addresses
        (r"\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Strasse|Str\.?|Weg|Platz)\b", "STREET_ADDRESS"),
    ],
    PIIType.CUSTOM_GERMAN: [
        (r"\b(Herr|Frau|Dr\.|Prof\.)\s+[A-Z\u00c0-\u00d6\u00d8-\u00de][a-z\u00c0-\u00ff]+\b", "DE_SALUTATION"),
    ],
}

# Obfuscation-aware patterns (Cat-2 adversarial defense)
OBFUSCATION_PATTERNS = [
    (r"\b[A-Z][0-9][a-z]{2}[0-9]\s+[A-Z][a-z]{1,2}[0-9][a-z]{2}[0-9]\b", "LEETSPEAK_NAME"),
    (r"\b\w+\s+dot\s+\w+\s+at\s+\w+\s+dot\s+\w+\b", "DOT_EMAIL"),
    (r"\b\w+\s+punkt\s+\w+\s+bei\s+\w+\s+punkt\s+\w+\b", "DE_DOT_EMAIL"),
    (r"\bzero[-\s]?\w+[-\s]?\w+[-\s]?\w+\b", "SPOKEN_PHONE"),
]

# Injection attack patterns (Cat-1 adversarial defense)
INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|rules)",
    r"(system|admin|developer|debug)\s+(prompt|mode|override|access)",
    r"disable\s+(pii|privacy|filter|safety)",
    r"(send|forward|exfiltrate|transmit)\s+(to|data)",
    r"cloud[_\s]override",
    r"\[SYSTEM[:\s]",
    r"jinja[_\s]inject",
    r"\{\{.*\}\}",  # template injection
]

# Residual PII patterns — second pass after masking
RESIDUAL_PATTERNS = [
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",   # email
    r"\b\d{10,}\b",                                              # long number sequences
    r"\b4[0-9]{3}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b",  # Visa card
    r"\b5[1-5][0-9]{2}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}\b",  # Mastercard
    r"\bDE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b",   # IBAN
    r"\b(\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b",  # phone
]


class PresidioPIIDetector(IPIIDetector):
    """
    Production PII detector combining:
    1. Microsoft Presidio (NER-based, 20+ entity types)
    2. German-specific regex patterns
    3. Credit card / IBAN / phone extended patterns
    4. Obfuscation-aware patterns for adversarial robustness
    5. Prompt injection detection
    """

    def __init__(self, threshold: float = 0.35, block_on_uncertainty: bool = True) -> None:
        # Lowered threshold from 0.6 -> 0.35 to catch more PII with fewer misses
        self._threshold = threshold
        self._block_on_uncertainty = block_on_uncertainty
        self._analyzer = self._init_presidio()
        self._injection_re = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]
        self._residual_re = [
            re.compile(p, re.IGNORECASE) for p in RESIDUAL_PATTERNS
        ]

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
            logger.warning("Presidio init failed (%s) — falling back to regex only", e)
            return None

    def detect_and_mask(self, query: Query) -> SanitizedQuery:
        text = query.raw_text
        masks: list[PIIMask] = []

        # 1. Check for prompt injection first — hard block
        self._check_injection(text, query)

        # 2. Run Presidio NER detection
        if self._analyzer:
            masks.extend(self._run_presidio(text))

        # 3. Run German-specific + credit card regex patterns
        masks.extend(self._run_german_patterns(text))

        # 4. Run obfuscation-aware patterns
        masks.extend(self._run_obfuscation_patterns(text))

        # 5. Apply all masks to produce sanitized text
        sanitized = self._apply_masks(text, masks)
        pii_found = len(masks) > 0

        # 6. Final safety check — scan sanitized output for anything that slipped through
        if self._residual_pii_check(sanitized):
            logger.warning("Residual PII detected after masking for query %s", query.id)
            pii_found = True

        return SanitizedQuery(
            original_query=query,
            sanitized_text=sanitized,
            masks_applied=masks,
            pii_detected=pii_found,
            safe_for_cloud=not pii_found,
        )

    def scan_text(self, text: str) -> bool:
        """Quick scan — returns True if PII detected."""
        dummy = Query(raw_text=text)
        result = self.detect_and_mask(dummy)
        return result.pii_detected

    def _check_injection(self, text: str, query: Query) -> None:
        from domain.exceptions import PromptInjectionError
        for pattern in self._injection_re:
            if pattern.search(text):
                logger.error("INJECTION DETECTED in query %s: %s", query.id, pattern.pattern)
                raise PromptInjectionError(pattern.pattern)

    def _run_presidio(self, text: str) -> list[PIIMask]:
        masks = []
        try:
            results = self._analyzer.analyze(
                text=text,
                language="en",
                score_threshold=self._threshold,
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION",
                    "CREDIT_CARD", "IBAN_CODE", "NRP", "MEDICAL_LICENSE",
                    "DATE_TIME", "IP_ADDRESS", "URL", "US_SSN",
                ],
            )
            for r in results:
                pii_type = self._presidio_to_pii_type(r.entity_type)
                masks.append(PIIMask(
                    original=text[r.start:r.end],
                    masked=f"[{r.entity_type}]",
                    pii_type=pii_type,
                    confidence=r.score,
                    position_start=r.start,
                    position_end=r.end,
                ))
        except Exception as e:
            logger.warning("Presidio analysis failed: %s", e)
        return masks

    def _run_german_patterns(self, text: str) -> list[PIIMask]:
        masks = []
        for pii_type, patterns in GERMAN_PATTERNS.items():
            for pattern_str, label in patterns:
                for match in re.finditer(pattern_str, text, re.IGNORECASE):
                    masks.append(PIIMask(
                        original=match.group(),
                        masked=f"[{label}]",
                        pii_type=pii_type,
                        confidence=0.95,
                        position_start=match.start(),
                        position_end=match.end(),
                    ))
        return masks

    def _run_obfuscation_patterns(self, text: str) -> list[PIIMask]:
        masks = []
        for pattern_str, label in OBFUSCATION_PATTERNS:
            for match in re.finditer(pattern_str, text, re.IGNORECASE):
                masks.append(PIIMask(
                    original=match.group(),
                    masked=f"[OBFUSCATED_{label}]",
                    pii_type=PIIType.NAME,
                    confidence=0.75,
                    position_start=match.start(),
                    position_end=match.end(),
                ))
        return masks

    @staticmethod
    def _apply_masks(text: str, masks: list[PIIMask]) -> str:
        if not masks:
            return text
        # Deduplicate overlapping masks — keep highest confidence
        sorted_masks = sorted(masks, key=lambda m: m.position_start)
        deduped = []
        last_end = -1
        for mask in sorted_masks:
            if mask.position_start >= last_end:
                deduped.append(mask)
                last_end = mask.position_end
            else:
                # Overlapping — keep the one with higher confidence
                if deduped and mask.confidence > deduped[-1].confidence:
                    deduped[-1] = mask

        # Apply in reverse order to preserve positions
        for mask in sorted(deduped, key=lambda m: m.position_start, reverse=True):
            text = text[:mask.position_start] + mask.masked + text[mask.position_end:]
        return text

    def _residual_pii_check(self, text: str) -> bool:
        """Second-pass check on masked output for anything that slipped through."""
        return any(p.search(text) for p in self._residual_re)

    @staticmethod
    def _presidio_to_pii_type(entity_type: str) -> PIIType:
        mapping = {
            "PERSON": PIIType.NAME,
            "EMAIL_ADDRESS": PIIType.EMAIL,
            "PHONE_NUMBER": PIIType.PHONE,
            "LOCATION": PIIType.ADDRESS,
            "CREDIT_CARD": PIIType.FINANCIAL,
            "IBAN_CODE": PIIType.FINANCIAL,
            "NRP": PIIType.ID_NUMBER,
            "MEDICAL_LICENSE": PIIType.MEDICAL,
            "DATE_TIME": PIIType.CUSTOM_GERMAN,
            "IP_ADDRESS": PIIType.CUSTOM_GERMAN,
            "URL": PIIType.CUSTOM_GERMAN,
            "US_SSN": PIIType.ID_NUMBER,
        }
        return mapping.get(entity_type, PIIType.CUSTOM_GERMAN)
