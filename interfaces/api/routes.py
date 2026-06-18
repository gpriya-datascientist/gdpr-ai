"""
Layer: INTERFACES
Imports allowed: all layers
Purpose: FastAPI routes — thin layer, delegates everything to orchestrator.
"""
import logging
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.container import build_document_parser, build_chunker, build_orchestrator, build_rag_service, get_settings
from domain.exceptions import EuroSecError, PIILeakError, PromptInjectionError
from domain.models import Query

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    text: str
    session_id: str | None = None
    document_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    route_taken: str
    provider_used: str
    pii_masked_count: int
    latency_ms: float
    audit_id: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str


class GDPRErasureResponse(BaseModel):
    session_id: str
    entries_deleted: int
    message: str


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Main query endpoint — routes through full orchestration pipeline."""
    settings = get_settings()
    orchestrator = build_orchestrator()

    query_obj = Query(
        raw_text=request.text,
        session_id=request.session_id,
        document_id=UUID(request.document_id) if request.document_id else None,
    )

    try:
        result = orchestrator.process(query_obj)
        return QueryResponse(
            answer=result.answer,
            route_taken=result.route_taken.value,
            provider_used=result.provider_used,
            pii_masked_count=result.pii_masked_count,
            latency_ms=round(result.latency_ms, 2),
            audit_id=str(result.audit_id),
        )
    except PromptInjectionError as e:
        logger.error("Injection attempt blocked: %s", e)
        raise HTTPException(status_code=400, detail="Query blocked: security violation")
    except EuroSecError as e:
        logger.error("EuroSecError: %s", e)
        raise HTTPException(status_code=500, detail="Internal processing error")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a document into the local vector store."""
    import tempfile, os
    settings = get_settings()

    # Save upload to temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parser = build_document_parser()
        chunker = build_chunker(settings)
        rag = build_rag_service(settings)

        document = parser.parse(tmp_path)
        document.filename = file.filename
        chunks = chunker.chunk(document)
        rag.ingest_document(document, chunks)

        return UploadResponse(
            document_id=str(document.id),
            filename=file.filename,
            chunks_created=len(chunks),
            message="Document ingested successfully — stored locally",
        )
    finally:
        os.unlink(tmp_path)


@router.delete("/gdpr/erase/{session_id}", response_model=GDPRErasureResponse)
async def gdpr_erase(session_id: str):
    """
    GDPR Article 17 — Right to erasure.
    Deletes all audit entries for a session.
    """
    from config.container import build_audit_logger
    settings = get_settings()
    audit = build_audit_logger(settings)
    count = audit.delete_by_session(session_id)
    return GDPRErasureResponse(
        session_id=session_id,
        entries_deleted=count,
        message=f"GDPR erasure complete: {count} records deleted",
    )


@router.get("/audit/{session_id}")
async def get_audit_log(session_id: str):
    """Return audit trail for a session — GDPR transparency."""
    from config.container import build_audit_logger
    settings = get_settings()
    audit = build_audit_logger(settings)
    entries = audit.get_entries(session_id)
    return {
        "session_id": session_id,
        "total_entries": len(entries),
        "entries": [
            {
                "id": str(e.id),
                "route": e.route_decision.value,
                "sensitivity": e.sensitivity_level.value,
                "pii_detected": e.pii_detected,
                "provider": e.provider_called,
                "gdpr_compliant": e.gdpr_compliant,
                "timestamp": e.created_at.isoformat(),
            }
            for e in entries
        ],
    }


@router.get("/health")
async def health():
    return {"status": "ok", "service": "EuroSec AI", "local_only_mode": get_settings().cloud_provider.value == "none"}
