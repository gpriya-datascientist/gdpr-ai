"""
Layer: DOMAIN
Imports allowed: stdlib only
Purpose: Core data models — no external dependencies ever
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class SensitivityLevel(Enum):
    HIGH = "high"        # contains PII / confidential doc content
    MEDIUM = "medium"    # potentially sensitive, needs review
    LOW = "low"          # general knowledge, safe for cloud


class RouteDecision(Enum):
    LOCAL_ONLY = "local_only"          # sensitive → stays on machine
    CLOUD_SANITIZED = "cloud_sanitized"  # general → sanitized cloud call
    BLOCKED = "blocked"                # injection detected → reject


class PIIType(Enum):
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    ID_NUMBER = "id_number"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    CUSTOM_GERMAN = "custom_german"    # Steuernummer, Personalausweis, IBAN


@dataclass
class PIIMask:
    original: str
    masked: str
    pii_type: PIIType
    confidence: float
    position_start: int
    position_end: int


@dataclass
class Document:
    id: UUID = field(default_factory=uuid4)
    filename: str = ""
    content: str = ""
    chunks: list[str] = field(default_factory=list)
    sensitivity: SensitivityLevel = SensitivityLevel.HIGH
    created_at: datetime = field(default_factory=datetime.utcnow)
    encrypted: bool = True


@dataclass
class Query:
    id: UUID = field(default_factory=uuid4)
    raw_text: str = ""
    document_id: Optional[UUID] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SanitizedQuery:
    original_query: Query
    sanitized_text: str
    masks_applied: list[PIIMask] = field(default_factory=list)
    pii_detected: bool = False
    safe_for_cloud: bool = False


@dataclass
class ClassifiedQuery:
    query: Query
    sensitivity: SensitivityLevel
    route: RouteDecision
    confidence: float
    reasoning: str


@dataclass
class RetrievedContext:
    chunks: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    source_document_id: Optional[UUID] = None


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str          # "local" | "groq" | "openai"
    tokens_used: int = 0
    latency_ms: float = 0.0
    cached: bool = False


@dataclass
class OrchestratorResponse:
    query_id: UUID
    answer: str
    route_taken: RouteDecision
    provider_used: str
    pii_masked_count: int
    latency_ms: float
    audit_id: UUID
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AuditEntry:
    id: UUID = field(default_factory=uuid4)
    query_id: UUID = field(default_factory=uuid4)
    session_id: Optional[str] = None
    route_decision: RouteDecision = RouteDecision.LOCAL_ONLY
    sensitivity_level: SensitivityLevel = SensitivityLevel.HIGH
    pii_detected: bool = False
    pii_count: int = 0
    provider_called: str = "local"
    gdpr_compliant: bool = True
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
