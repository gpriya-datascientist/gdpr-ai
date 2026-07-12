"""
Layer: INDUSTRIAL - Visual Inspection
Purpose: Defect detection engine with precise good/bad differentiation.
         Uses local Ollama vision first, falls back to Gemini if confidence is low.
"""
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

LOCAL_CONFIDENCE_THRESHOLD = 0.45
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

INSPECTION_PROMPT_TEMPLATE = """You are an expert visual quality inspector for BSH industrial metal press tools.

Inspection object: {part_name}.
{description}

You are looking at grayscale close-up photos of curved metal press tool surfaces.

GOOD parts look like:
- Smooth continuously curved metal surface with flowing lines
- Bright white light reflections or glare on the metal -- this is NORMAL and expected
- Parallel tooling marks from machining -- these are NORMAL
- Clean smooth edges and curves with no breaks in the metal

ANOMALY (defect) parts show:
- A DARK irregular crack, tear, hole or opening that BREAKS the smooth metal surface
- A jagged dark line running across or through the metal where it has split
- A visible gap or rupture in the metal surface
- Any discontinuity that looks like a fracture or tear

KEY RULE: Bright white areas = light reflection = GOOD. Dark breaks in the metal = crack = ANOMALY.
Do NOT mark bright or white areas as defects.
Only dark irregular breaks or fractures in the metal surface are defects.

Also identify which part is affected:
- door_left: left door panel or press tool
- door_right: right door panel
- hood: hood panel
- unknown: cannot determine

If ANOMALY found, estimate bounding box (0.0=top-left, 1.0=bottom-right).

Respond ONLY with valid JSON:
{{"verdict": "ANOMALY" or "GOOD", "confidence": 0.95, "reason": "specific description max 50 words", "defect_type": "crack or deformation or surface_gap or misalignment or weld_defect or scratch or unknown", "affected_part": "door_left or door_right or hood or unknown", "box": {{"x_min": 0.1, "y_min": 0.2, "x_max": 0.6, "y_max": 0.8}} or null}}"""

DEFECT_CONTROL_MAP = {
    ("crack",        "door_left"):  {"param": "left_door_stroke_rate",  "action": "reduce",   "pct": 15, "reason": "Crack on left door -- reduce stroke rate to relieve pressure"},
    ("crack",        "door_right"): {"param": "right_door_stroke_rate", "action": "reduce",   "pct": 15, "reason": "Crack on right door -- reduce stroke rate"},
    ("crack",        "hood"):       {"param": "hood_stroke_rate",       "action": "reduce",   "pct": 15, "reason": "Crack on hood -- reduce hood stroke rate"},
    ("crack",        "unknown"):    {"param": "left_door_stroke_rate",  "action": "reduce",   "pct": 12, "reason": "Crack detected -- reduce stroke rate as precaution"},
    ("deformation",  "door_left"):  {"param": "left_door_stroke_rate",  "action": "reduce",   "pct": 10, "reason": "Deformation on left door -- reduce stroke force"},
    ("deformation",  "door_right"): {"param": "right_door_stroke_rate", "action": "reduce",   "pct": 10, "reason": "Deformation on right door -- reduce stroke force"},
    ("deformation",  "hood"):       {"param": "hood_stroke_rate",       "action": "reduce",   "pct": 10, "reason": "Deformation on hood"},
    ("deformation",  "unknown"):    {"param": "torque_nm",              "action": "reduce",   "pct": 10, "reason": "Deformation -- reduce torque"},
    ("surface_gap",  "door_left"):  {"param": "left_door_stroke_rate",  "action": "reduce",   "pct": 12, "reason": "Surface gap on left door"},
    ("surface_gap",  "door_right"): {"param": "right_door_stroke_rate", "action": "reduce",   "pct": 12, "reason": "Surface gap on right door"},
    ("surface_gap",  "hood"):       {"param": "hood_stroke_rate",       "action": "reduce",   "pct": 12, "reason": "Surface gap on hood"},
    ("surface_gap",  "unknown"):    {"param": "cycle_time_s",           "action": "increase", "pct": 10, "reason": "Surface gap -- increase cycle time"},
    ("misalignment", "door_left"):  {"param": "cycle_time_s",           "action": "increase", "pct": 12, "reason": "Misalignment on left door"},
    ("misalignment", "door_right"): {"param": "cycle_time_s",           "action": "increase", "pct": 12, "reason": "Misalignment on right door"},
    ("misalignment", "hood"):       {"param": "cycle_time_s",           "action": "increase", "pct": 10, "reason": "Hood misalignment"},
    ("misalignment", "unknown"):    {"param": "cycle_time_s",           "action": "increase", "pct": 10, "reason": "Misalignment -- slow down cycle"},
    ("weld_defect",  "door_left"):  {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 10, "reason": "Weld defect on left door"},
    ("weld_defect",  "door_right"): {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 10, "reason": "Weld defect on right door"},
    ("weld_defect",  "hood"):       {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 8,  "reason": "Weld defect on hood"},
    ("weld_defect",  "unknown"):    {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 10, "reason": "Weld defect"},
    ("scratch",      "door_left"):  {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 8,  "reason": "Scratch on left door"},
    ("scratch",      "door_right"): {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 8,  "reason": "Scratch on right door"},
    ("scratch",      "hood"):       {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 8,  "reason": "Scratch on hood"},
    ("scratch",      "unknown"):    {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 8,  "reason": "Scratch detected"},
    ("unknown",      "unknown"):    {"param": "assembly_speed_ms",      "action": "reduce",   "pct": 8,  "reason": "Unknown defect -- reduce assembly speed"},
}


@dataclass
class InspectionResult:
    verdict: str
    confidence: float
    reason: str
    defect_type: Optional[str]
    backend_used: str
    latency_ms: float
    boxes: list = field(default_factory=list)
    affected_part: Optional[str] = None
    control_suggestion: Optional[dict] = None
    raw_response: str = ""


class VisualInspectionEngine:
    def __init__(self, ollama_base_url="http://localhost:11434", ollama_vision_model="llava:7b",
                 gemini_api_key=None, gemini_model="gemini-2.0-flash"):
        self._ollama_url = ollama_base_url.rstrip("/")
        self._ollama_model = ollama_vision_model
        self._gemini_key = gemini_api_key
        self._gemini_model = gemini_model

    async def inspect(self, image_bytes, part_name="BMW Manufacturing Part",
                      description="Check for cracks, deformations, surface gaps, misalignment, weld defects or scratches.",
                      mime="image/jpeg"):
        prompt = INSPECTION_PROMPT_TEMPLATE.format(part_name=part_name, description=description)
        local_result = await self._try_local(image_bytes, prompt, mime)
        if local_result and local_result.confidence >= LOCAL_CONFIDENCE_THRESHOLD:
            logger.info("Local model confident (%.2f)", local_result.confidence)
            local_result.control_suggestion = self._get_control_suggestion(local_result)
            return local_result
        if self._gemini_key:
            gemini_result = await self._try_gemini(image_bytes, prompt, mime)
            if gemini_result:
                gemini_result.control_suggestion = self._get_control_suggestion(gemini_result)
                return gemini_result
        if local_result:
            local_result.control_suggestion = self._get_control_suggestion(local_result)
            return local_result
        return InspectionResult(
            verdict="ERROR", confidence=0.0,
            reason="All vision backends unavailable. Ensure Ollama is running or Gemini API key is set.",
            defect_type=None, backend_used="none", latency_ms=0.0)

    async def _try_local(self, image_bytes, prompt, mime):
        start = time.perf_counter()
        try:
            img_b64 = base64.b64encode(image_bytes).decode()
            payload = {"model": self._ollama_model, "prompt": prompt,
                       "images": [img_b64], "stream": False, "options": {"temperature": 0.0}}
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(f"{self._ollama_url}/api/generate", json=payload)
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()
            return self._parse_response(raw, "local", (time.perf_counter()-start)*1000)
        except Exception as e:
            logger.warning("Local vision failed: %s", e)
            return None

    async def _try_gemini(self, image_bytes, prompt, mime):
        start = time.perf_counter()
        try:
            img_b64 = base64.b64encode(image_bytes).decode()
            model = self._gemini_model.replace("models/", "")
            url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
            payload = {
                "contents": [{"parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": img_b64}}
                ]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}
            }
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=payload, params={"key": self._gemini_key})
                resp.raise_for_status()
                raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            return self._parse_response(raw, "gemini", (time.perf_counter()-start)*1000)
        except Exception as e:
            logger.warning("Gemini vision failed: %s", e)
            return None

    @staticmethod
    def _parse_response(raw, backend, latency_ms):
        try:
            clean = raw.strip()
            # Strip markdown code fences (handle leading spaces too)
            if "```" in clean:
                parts = clean.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if "{" in part and "verdict" in part:
                        clean = part
                        break

            # Try to extract JSON object from anywhere in the response
            import re
            json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', clean, re.DOTALL)
            if json_match:
                clean = json_match.group(0)

            parsed = json.loads(clean)
            verdict = str(parsed.get("verdict", "GOOD")).upper()
            if verdict not in ("GOOD", "ANOMALY"):
                verdict = "GOOD"
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            reason = str(parsed.get("reason", ""))[:300]
            defect_type = parsed.get("defect_type")
            if isinstance(defect_type, list):
                defect_type = defect_type[0] if defect_type else None
            if defect_type in ("null", "", None):
                defect_type = None
            if defect_type:
                defect_type = str(defect_type).lower()
            affected_part = parsed.get("affected_part", "unknown")
            if isinstance(affected_part, list):
                affected_part = affected_part[0] if affected_part else "unknown"
            if affected_part not in ("door_left", "door_right", "hood"):
                affected_part = "unknown"
            boxes = []
            box = parsed.get("box")
            if box and isinstance(box, dict) and verdict == "ANOMALY":
                try:
                    boxes = [{"x_min": float(box.get("x_min", 0.1)), "y_min": float(box.get("y_min", 0.3)),
                              "x_max": float(box.get("x_max", 0.7)), "y_max": float(box.get("y_max", 0.8))}]
                except Exception:
                    boxes = []
            elif isinstance(box, list) and len(box) > 0 and verdict == "ANOMALY":
                try:
                    b = box[0] if isinstance(box[0], dict) else {}
                    boxes = [{"x_min": float(b.get("x_min", 0.1)), "y_min": float(b.get("y_min", 0.3)),
                              "x_max": float(b.get("x_max", 0.7)), "y_max": float(b.get("y_max", 0.8))}]
                except Exception:
                    boxes = []
            return InspectionResult(
                verdict=verdict, confidence=confidence, reason=reason,
                defect_type=defect_type, affected_part=affected_part,
                backend_used=backend, latency_ms=latency_ms,
                boxes=boxes, raw_response=raw[:500])
        except Exception as e:
            logger.warning("Failed to parse: %s | raw: %s", e, raw[:300])
            # Last resort: keyword scan
            raw_upper = raw.upper()
            if "ANOMALY" in raw_upper:
                return InspectionResult(verdict="ANOMALY", confidence=0.6,
                    reason="Defect detected", defect_type="unknown",
                    affected_part="unknown", backend_used=backend, latency_ms=latency_ms)
            return InspectionResult(verdict="GOOD", confidence=0.5,
                reason="No defect detected", defect_type=None,
                affected_part="unknown", backend_used=backend, latency_ms=latency_ms)

    @staticmethod
    def _get_control_suggestion(result):
        if result.verdict != "ANOMALY":
            return None
        defect = (result.defect_type or "unknown").lower()
        part = (result.affected_part or "unknown").lower()
        mapping = (
            DEFECT_CONTROL_MAP.get((defect, part)) or
            DEFECT_CONTROL_MAP.get((defect, "unknown")) or
            DEFECT_CONTROL_MAP.get(("unknown", "unknown"))
        )
        if not mapping:
            return None
        param, action, pct = mapping["param"], mapping["action"], mapping["pct"]
        baselines = {
            "left_door_stroke_rate": 45.0, "right_door_stroke_rate": 44.0,
            "hood_stroke_rate": 30.0, "assembly_speed_ms": 2.8,
            "cycle_time_s": 240.0, "torque_nm": 150.0,
        }
        base = baselines.get(param, 1.0)
        mult = (1 - pct/100) if action == "reduce" else (1 + pct/100)
        new_val = round(base * mult, 2)
        param_label = param.replace("_", " ")
        cmd = f"{'Reduce' if action == 'reduce' else 'Increase'} {param_label} by {pct}%"
        return {
            "param": param, "param_label": param_label, "action": action, "pct": pct,
            "current_value": base, "suggested_value": new_val,
            "change": f"{'-' if action == 'reduce' else '+'}{pct}%",
            "reason": mapping["reason"], "defect_type": defect, "affected_part": part,
            "command": cmd, "risk": "medium" if pct > 12 else "low",
        }
