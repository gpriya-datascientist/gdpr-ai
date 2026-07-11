"""
Layer: INTERFACES — Visual Inspection API Routes
Purpose: FastAPI routes for visual defect detection.
         POST /api/v1/industrial/inspect — upload image, get GOOD/ANOMALY verdict + control suggestion.
"""
import logging
import os
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Optional

from industrial.vision.inspector import VisualInspectionEngine, InspectionResult

logger = logging.getLogger(__name__)
vision_router = APIRouter()

# Initialize engine from environment
_engine = VisualInspectionEngine(
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ollama_vision_model=os.getenv("OLLAMA_VISION_MODEL", "llava:7b"),
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB


class InspectResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str
    defect_type: Optional[str]
    backend_used: str
    latency_ms: float
    control_suggestion: Optional[dict]


class ControlApplyRequest(BaseModel):
    param: str
    action: str
    pct: float
    current_value: float
    suggested_value: float


@vision_router.post("/inspect", response_model=InspectResponse)
async def inspect_image(
    file: UploadFile = File(...),
    part_name: str = "BMW Manufacturing Part",
    description: str = "Check for cracks, deformations, surface gaps, misalignment, weld defects or scratches.",
):
    """
    Inspect an uploaded image for manufacturing defects.
    Returns verdict (GOOD/ANOMALY), confidence, defect type, and Control Plane suggestion.
    """
    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 20MB)")

    mime = file.content_type or "image/jpeg"
    if not mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        result = await _engine.inspect(
            image_bytes=data,
            part_name=part_name,
            description=description,
            mime=mime,
        )
    except Exception as e:
        logger.error("Inspection failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")

    return InspectResponse(
        verdict=result.verdict,
        confidence=result.confidence,
        reason=result.reason,
        defect_type=result.defect_type,
        backend_used=result.backend_used,
        latency_ms=round(result.latency_ms, 1),
        control_suggestion=result.control_suggestion,
    )


@vision_router.get("/inspect/health")
async def inspection_health():
    """Check which vision backends are available."""
    ollama_ok = False
    gemini_ok = bool(os.getenv("GEMINI_API_KEY"))

    try:
        import httpx
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{_engine._ollama_url}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                ollama_ok = any("llava" in m for m in models)
    except Exception:
        pass

    return {
        "status": "ok",
        "backends": {
            "local_ollama": {"available": ollama_ok, "model": _engine._ollama_model},
            "gemini": {"available": gemini_ok, "model": _engine._gemini_model},
        },
        "strategy": "local_first_gemini_fallback",
    }


@vision_router.get("/inspect/samples")
async def get_sample_info():
    """Return info about available sample images for testing."""
    return {
        "dataset": "BSH Industrial Press Tool Dataset",
        "description": "Metal press tool inspection images — cracks and surface deformations",
        "total_images": 24,
        "good_images": 12,
        "bad_images": 12,
        "sets": ["Set_1", "Set_2", "Set_3", "Set_6"],
        "defect_types": ["crack", "deformation", "surface_gap"],
        "usage": "Upload any image to POST /api/v1/industrial/inspect",
    }
