"""
Layer: INDUSTRIAL - Visual Inspection
Purpose: Defect detection using Gemini (primary) + llava fallback.
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

INSPECTION_PROMPT_TEMPLATE = """You are a strict visual quality inspector in an industrial press shop.
Task: inspect ONE image for defects in a manufacturing part.

Inspection object: {part_name}.
{description}

Definitions:
- GOOD: the part surface looks normal. Smooth continuous curves, bright white glare/reflections, parallel machining lines, dark background fixture, shadows at edges — all NORMAL.
- ANOMALY: a dark irregular crack, fracture, tear or rupture that visibly BREAKS the smooth continuous metal surface.

Be conservative: only report ANOMALY when a dark break in the metal surface is clearly visible.
Bright white areas = light reflection = NOT a defect.

Respond ONLY with valid JSON:
{{"verdict": "ANOMALY" or "GOOD", "confidence": <float 0.0-1.0>, "reason": "<brief max 30 words>", "defect_type": "<crack or deformation or unknown or null>", "affected_part": "<door_left or door_right or hood or unknown>", "box": {{"x_min": 0.1, "y_min": 0.3, "x_max": 0.7, "y_max": 0.8}} or null}}"""


DEFECT_CONTROL_MAP = {
    ("crack","door_left"):   {"param":"left_door_stroke_rate","action":"reduce","pct":15,"reason":"Crack on left door -- reduce stroke rate"},
    ("crack","door_right"):  {"param":"right_door_stroke_rate","action":"reduce","pct":15,"reason":"Crack on right door"},
    ("crack","hood"):        {"param":"hood_stroke_rate","action":"reduce","pct":15,"reason":"Crack on hood"},
    ("crack","unknown"):     {"param":"left_door_stroke_rate","action":"reduce","pct":12,"reason":"Crack detected -- reduce stroke rate"},
    ("deformation","door_left"): {"param":"left_door_stroke_rate","action":"reduce","pct":10,"reason":"Deformation on left door"},
    ("deformation","door_right"):{"param":"right_door_stroke_rate","action":"reduce","pct":10,"reason":"Deformation on right door"},
    ("deformation","unknown"):   {"param":"torque_nm","action":"reduce","pct":10,"reason":"Deformation -- reduce torque"},
    ("unknown","unknown"):       {"param":"assembly_speed_ms","action":"reduce","pct":8,"reason":"Unknown defect -- reduce speed"},
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
    def __init__(self, ollama_base_url="http://localhost:11434",
                 ollama_vision_model="llava:13b",
                 gemini_api_key=None, gemini_model="gemini-2.0-flash"):
        self._ollama_url = ollama_base_url.rstrip("/")
        self._ollama_model = ollama_vision_model
        self._gemini_key = gemini_api_key
        self._gemini_model = gemini_model

    async def inspect(self, image_bytes, part_name="BMW Manufacturing Part",
                      description="Check for cracks, fractures, or tears in the metal surface.",
                      mime="image/jpeg", force_gemini=False):
        prompt = INSPECTION_PROMPT_TEMPLATE.format(
            part_name=part_name, description=description)

        # Try local first — fast and no quota limits
        r = await self._try_local(image_bytes, prompt, mime)
        if r and r.verdict in ("GOOD", "ANOMALY"):
            r.control_suggestion = self._get_control_suggestion(r)
            return r

        # Fallback to Gemini only if local fails
        if self._gemini_key:
            r = await self._try_gemini(image_bytes, prompt, mime)
            if r and r.verdict in ("GOOD", "ANOMALY"):
                r.control_suggestion = self._get_control_suggestion(r)
                return r

        return InspectionResult(verdict="ERROR", confidence=0.0,
            reason="All backends unavailable.", defect_type=None,
            backend_used="none", latency_ms=0.0)


    async def _try_local(self, image_bytes, prompt, mime):
        start = time.perf_counter()
        try:
            img_b64 = base64.b64encode(image_bytes).decode()
            payload = {"model": self._ollama_model, "prompt": prompt,
                       "images": [img_b64], "stream": False,
                       "options": {"temperature": 0.0}}
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{self._ollama_url}/api/generate", json=payload)
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()
            return self._parse_response(raw, "local",
                                        (time.perf_counter()-start)*1000)
        except Exception as e:
            logger.warning("Local vision failed: %s", e)
            return None

    async def _try_gemini(self, image_bytes, prompt, mime):
        start = time.perf_counter()
        try:
            img_b64 = base64.b64encode(image_bytes).decode()
            model = self._gemini_model.replace("models/", "")
            url = (f"https://generativelanguage.googleapis.com"
                   f"/v1/models/{model}:generateContent")
            payload = {
                "contents": [{"parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": img_b64}}
                ]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 300}
            }
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=payload,
                                         params={"key": self._gemini_key})
                resp.raise_for_status()
                raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            return self._parse_response(raw, "gemini",
                                        (time.perf_counter()-start)*1000)
        except Exception as e:
            logger.warning("Gemini vision failed: %s", e)
            return None

    @staticmethod
    def _parse_response(raw, backend, latency_ms):
        try:
            import re
            clean = raw.strip()
            if "```" in clean:
                for p in clean.split("```"):
                    p = p.strip()
                    if p.startswith("json"):
                        p = p[4:].strip()
                    if '"verdict"' in p:
                        clean = p
                        break
            m = re.search(r'\{.*?"verdict".*?\}', clean, re.DOTALL)
            if m:
                clean = m.group(0)
            parsed = json.loads(clean)
            verdict = str(parsed.get("verdict", "GOOD")).upper()
            if verdict not in ("GOOD", "ANOMALY"):
                verdict = "GOOD"
            conf = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            reason = str(parsed.get("reason", ""))[:300]
            dt = parsed.get("defect_type")
            if isinstance(dt, list): dt = dt[0] if dt else None
            if dt in ("null", "", None): dt = None
            if dt: dt = str(dt).lower()
            ap = str(parsed.get("affected_part", "unknown")).lower()
            if ap not in ("door_left", "door_right", "hood"):
                ap = "unknown"
            boxes = []
            box = parsed.get("box")
            if box and isinstance(box, dict) and verdict == "ANOMALY":
                try:
                    boxes = [{"x_min": float(box.get("x_min", 0.1)),
                              "y_min": float(box.get("y_min", 0.3)),
                              "x_max": float(box.get("x_max", 0.7)),
                              "y_max": float(box.get("y_max", 0.8))}]
                except Exception:
                    pass
            return InspectionResult(verdict=verdict, confidence=conf,
                reason=reason, defect_type=dt, affected_part=ap,
                backend_used=backend, latency_ms=latency_ms, boxes=boxes,
                raw_response=raw[:500])
        except Exception as e:
            logger.warning("Parse failed: %s | raw: %s", e, raw[:200])
            if "ANOMALY" in raw.upper():
                return InspectionResult(verdict="ANOMALY", confidence=0.6,
                    reason="Defect detected", defect_type="unknown",
                    affected_part="unknown", backend_used=backend,
                    latency_ms=latency_ms)
            return InspectionResult(verdict="GOOD", confidence=0.5,
                reason="No defect detected", defect_type=None,
                affected_part="unknown", backend_used=backend,
                latency_ms=latency_ms)

    @staticmethod
    def _get_control_suggestion(result):
        if result.verdict != "ANOMALY":
            return None
        defect = (result.defect_type or "unknown").lower()
        part = (result.affected_part or "unknown").lower()
        mapping = (DEFECT_CONTROL_MAP.get((defect, part)) or
                   DEFECT_CONTROL_MAP.get((defect, "unknown")) or
                   DEFECT_CONTROL_MAP.get(("unknown", "unknown")))
        if not mapping:
            return None
        param, action, pct = mapping["param"], mapping["action"], mapping["pct"]
        bases = {"left_door_stroke_rate": 45.0, "right_door_stroke_rate": 44.0,
                 "hood_stroke_rate": 30.0, "assembly_speed_ms": 2.8,
                 "cycle_time_s": 240.0, "torque_nm": 150.0}
        base = bases.get(param, 1.0)
        mult = (1-pct/100) if action == "reduce" else (1+pct/100)
        new_val = round(base * mult, 2)
        label = param.replace("_", " ")
        cmd = f"{'Reduce' if action=='reduce' else 'Increase'} {label} by {pct}%"
        return {"param": param, "param_label": label, "action": action,
                "pct": pct, "current_value": base, "suggested_value": new_val,
                "change": f"{'-' if action=='reduce' else '+'}{pct}%",
                "reason": mapping["reason"], "defect_type": defect,
                "affected_part": part, "command": cmd,
                "risk": "medium" if pct > 12 else "low"}
