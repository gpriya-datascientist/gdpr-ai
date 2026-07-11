"""
Layer: INDUSTRIAL — Visual Inspection
Purpose: Defect detection engine — inspects images for manufacturing defects.
         Uses local Ollama vision first, falls back to Gemini if confidence is low.
         Adapted from PressVisionLoop (AGPL-3.0).
"""
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Detection thresholds
LOCAL_CONFIDENCE_THRESHOLD = 0.45   # lowered — catch more defects
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Inspection prompt template (adapted from PressVisionLoop SEED_TEMPLATE)
INSPECTION_PROMPT_TEMPLATE = """You are a strict visual quality inspector in an industrial metal press shop.
Task: inspect this image for ANY manufacturing defect.

Inspection object: {part_name}.
{description}

Look carefully for:
- Cracks: any thin dark line, fracture, or split in the metal surface
- Deformations: bent, dented, or warped areas
- Surface gaps: unexpected openings, notches, or holes not part of the design
- Misalignment: parts that are not properly aligned
- Weld defects: irregular weld beads, porosity, or gaps
- Scratches: deep surface marks

IMPORTANT: These are industrial parts — even small cracks or notches are ANOMALIES.
Sharp angular cuts, unusual notches, or breaks in the smooth metal surface are defects.
Do NOT classify a visible crack or notch as GOOD.

If ANOMALY, estimate bounding box coordinates (0.0=top-left, 1.0=bottom-right):
x_min, y_min = top-left of defect
x_max, y_max = bottom-right of defect

Respond ONLY with valid JSON:
{{"verdict": "ANOMALY" or "GOOD", "confidence": <float 0.0-1.0>, "reason": "<specific description of what you see, max 50 words>", "defect_type": "<crack|deformation|surface_gap|misalignment|weld_defect|scratch|unknown or null>", "box": {{"x_min": 0.1, "y_min": 0.2, "x_max": 0.6, "y_max": 0.8}} or null}}"""

# Defect → Control Plane parameter mapping
DEFECT_CONTROL_MAP = {
    "crack":         {"param": "left_door_stroke_rate",  "action": "reduce",   "pct": 15, "reason": "Crack indicates over-pressure on press tool"},
    "deformation":   {"param": "torque_nm",              "action": "reduce",   "pct": 10, "reason": "Deformation caused by excess torque"},
    "surface_gap":   {"param": "left_door_stroke_rate",  "action": "reduce",   "pct": 12, "reason": "Surface gap indicates stroke rate misalignment"},
    "misalignment":  {"param": "cycle_time_s",           "action": "increase", "pct": 12, "reason": "Misalignment needs slower cycle for precision"},
    "weld_defect":   {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 10, "reason": "Weld defect caused by excessive assembly speed"},
    "scratch":       {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 8,  "reason": "Scratch from high-speed contact — reduce speed"},
    "unknown":       {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 8,  "reason": "Unknown defect — reduce assembly speed as precaution"},
}


@dataclass
class InspectionResult:
    verdict: str            # "GOOD" or "ANOMALY"
    confidence: float
    reason: str
    defect_type: Optional[str]
    backend_used: str       # "local" or "gemini"
    latency_ms: float
    boxes: list = field(default_factory=list)
    control_suggestion: Optional[dict] = None
    raw_response: str = ""


class VisualInspectionEngine:
    """
    Inspects manufacturing images for defects.
    Strategy: try local Ollama vision first → fallback to Gemini if low confidence.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_vision_model: str = "llava:7b",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
    ):
        self._ollama_url = ollama_base_url.rstrip("/")
        self._ollama_model = ollama_vision_model
        self._gemini_key = gemini_api_key
        self._gemini_model = gemini_model

    async def inspect(
        self,
        image_bytes: bytes,
        part_name: str = "BMW Manufacturing Part",
        description: str = "Check for cracks, deformations, surface gaps, misalignment, weld defects or scratches.",
        mime: str = "image/jpeg",
    ) -> InspectionResult:
        """Inspect an image. Tries local first, falls back to Gemini."""
        prompt = INSPECTION_PROMPT_TEMPLATE.format(
            part_name=part_name,
            description=description,
        )

        # Step 1: Try local Ollama vision
        local_result = await self._try_local(image_bytes, prompt, mime)

        if local_result and local_result.confidence >= LOCAL_CONFIDENCE_THRESHOLD:
            logger.info("Visual inspection: local model confident (%.2f) — using local result", local_result.confidence)
            local_result.control_suggestion = self._get_control_suggestion(local_result)
            return local_result

        # Step 2: Fallback to Gemini
        if self._gemini_key:
            logger.info("Visual inspection: local confidence low (%.2f) — falling back to Gemini",
                       local_result.confidence if local_result else 0.0)
            gemini_result = await self._try_gemini(image_bytes, prompt, mime)
            if gemini_result:
                gemini_result.control_suggestion = self._get_control_suggestion(gemini_result)
                return gemini_result

        # Step 3: Return local result even if low confidence
        if local_result:
            local_result.control_suggestion = self._get_control_suggestion(local_result)
            return local_result

        # Step 4: All backends failed
        return InspectionResult(
            verdict="ERROR",
            confidence=0.0,
            reason="All vision backends unavailable. Ensure Ollama is running or Gemini API key is set.",
            defect_type=None,
            backend_used="none",
            latency_ms=0.0,
        )

    async def _try_local(self, image_bytes: bytes, prompt: str, mime: str) -> Optional[InspectionResult]:
        """Call local Ollama vision model (llava)."""
        start = time.perf_counter()
        try:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            payload = {
                "model": self._ollama_model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.0},
            }
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{self._ollama_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("response", "").strip()

            latency_ms = (time.perf_counter() - start) * 1000
            return self._parse_response(raw, "local", latency_ms)

        except Exception as e:
            logger.warning("Local vision model failed: %s", e)
            return None

    async def _try_gemini(self, image_bytes: bytes, prompt: str, mime: str) -> Optional[InspectionResult]:
        """Call Gemini vision API."""
        start = time.perf_counter()
        try:
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            # Use v1 API which supports newer models
            model = self._gemini_model.replace("models/", "")
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": mime, "data": img_b64}},
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 512,
                },
            }
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    params={"key": self._gemini_key},
                )
                resp.raise_for_status()
                data = resp.json()

            raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            latency_ms = (time.perf_counter() - start) * 1000
            return self._parse_response(raw, "gemini", latency_ms)

        except Exception as e:
            logger.warning("Gemini vision failed: %s", e)
            return None

    @staticmethod
    def _parse_response(raw: str, backend: str, latency_ms: float) -> Optional[InspectionResult]:
        """Parse JSON response from vision model."""
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            parsed = json.loads(clean)

            verdict = str(parsed.get("verdict", "GOOD")).upper()
            if verdict not in ("GOOD", "ANOMALY"):
                verdict = "GOOD"

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(parsed.get("reason", ""))[:300]
            defect_type = parsed.get("defect_type")
            if isinstance(defect_type, list):
                defect_type = defect_type[0] if defect_type else None
            if defect_type in ("null", "", None):
                defect_type = None
            if defect_type:
                defect_type = str(defect_type).lower()

            # Extract bounding box if provided
            boxes = []
            box = parsed.get("box")
            if box and isinstance(box, dict) and verdict == "ANOMALY":
                try:
                    boxes = [{
                        "x_min": float(box.get("x_min", 0.1)),
                        "y_min": float(box.get("y_min", 0.3)),
                        "x_max": float(box.get("x_max", 0.7)),
                        "y_max": float(box.get("y_max", 0.8)),
                    }]
                except Exception:
                    boxes = []
            elif isinstance(box, list) and len(box) > 0 and verdict == "ANOMALY":
                try:
                    b = box[0] if isinstance(box[0], dict) else {}
                    boxes = [{
                        "x_min": float(b.get("x_min", 0.1)),
                        "y_min": float(b.get("y_min", 0.3)),
                        "x_max": float(b.get("x_max", 0.7)),
                        "y_max": float(b.get("y_max", 0.8)),
                    }]
                except Exception:
                    boxes = []

            return InspectionResult(
                verdict=verdict,
                confidence=confidence,
                reason=reason,
                defect_type=defect_type,
                backend_used=backend,
                latency_ms=latency_ms,
                boxes=boxes,
                raw_response=raw[:500],
            )
        except Exception as e:
            logger.warning("Failed to parse vision response: %s | raw: %s", e, raw[:200])
            return None

    @staticmethod
    def _get_control_suggestion(result: InspectionResult) -> Optional[dict]:
        """Map defect type to Control Plane parameter suggestion."""
        if result.verdict != "ANOMALY":
            return None
        defect = result.defect_type or "unknown"
        mapping = DEFECT_CONTROL_MAP.get(defect, DEFECT_CONTROL_MAP["unknown"])
        param = mapping["param"]
        action = mapping["action"]
        pct = mapping["pct"]
        baselines = {
            "left_door_stroke_rate": 45.0,
            "right_door_stroke_rate": 44.0,
            "hood_stroke_rate": 30.0,
            "assembly_speed_ms": 2.8,
            "cycle_time_s": 240.0,
            "torque_nm": 150.0,
        }
        base = baselines.get(param, 1.0)
        mult = (1 - pct / 100) if action == "reduce" else (1 + pct / 100)
        new_val = round(base * mult, 2)
        return {
            "param": param,
            "param_label": param.replace("_", " "),
            "action": action,
            "pct": pct,
            "current_value": base,
            "suggested_value": new_val,
            "change": f"{'-' if action == 'reduce' else '+'}{pct}%",
            "reason": mapping["reason"],
            "defect_type": defect,
            "risk": "medium" if pct > 12 else "low",
        }
